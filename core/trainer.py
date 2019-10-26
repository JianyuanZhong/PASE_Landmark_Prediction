import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
import json
import pickle as pkl
import numpy as np
import io
import matplotlib.pyplot as plt
from PIL import Image
from tensorboardX import SummaryWriter
from tqdm import tqdm, trange
from pase.models.modules import Saver
from core.model.pase2landmark import pase2landmark, paseLSTMLandmark, featureLSTMLandmark
from core.lr_scheduler import LR_Scheduler
from core.losses import LandmarkLoss, RMSELoss


class Trainer(object):

    def __init__(self, PASE_cfg, MLP_cfg, trainloader, validloader, device, args):
        self.args = args
        self.trainloader = trainloader
        self.validloader = validloader
        self.device = device

        if self.args.model == 'MLP':
            self.model = pase2landmark(PASE_cfg, MLP_cfg, args.PASE_ckpt).to(device)
        elif self.args.model == 'LSTM':
            self.model = paseLSTMLandmark(PASE_cfg, MLP_cfg, args.PASE_ckpt).to(device)
        elif self.args.model == "feature":
            f, l, r = self.trainloader.dataset.__getitem__(0)
            input_dim = f.shape[0]
            self.model = featureLSTMLandmark(MLP_cfg, input_dim).to(device)
        
        iter_per_epoch = trainloader.dataset.__len__() // args.batch_size
        print("=" * 50)
        print("iter per epoch: {}".format(iter_per_epoch))

        if self.args.PASE_optim:
            self.pase_optim = optim.Adam(self.model.pase.parameters(), lr=args.PASE_lr)
            self.pase_scheduler = LR_Scheduler('poly', 'pase', self.args.PASE_lr, self.args.epoch, iter_per_epoch, lr_step=30)
        
        self.mlp_optim = optim.Adam(self.model.decoder.parameters(), lr=args.lr)
        self.mlp_scheduler = LR_Scheduler('poly', 'decoder', self.args.lr, self.args.epoch, iter_per_epoch, lr_step=30)

        self.Loss = LandmarkLoss()
        self.RMSELoss = RMSELoss()

        if self.args.PASE_optim:
            self.pase_saver = Saver(self.model.pase, self.args.save_path, max_ckpts=5, optimizer=self.pase_optim, prefix='PASE-')
        
        self.decoder_saver = Saver(self.model.decoder, self.args.save_path, max_ckpts=5, optimizer=self.mlp_optim, prefix='DECODER-')

        self.best_loss = np.Infinity

        self.writer = SummaryWriter(self.args.save_path)
        self.step = -1        


    def train(self):

        begin_epoch = self.resume_training()

        if self.args.early_stopping:
            self.early_stopping()

        for e in range(begin_epoch, self.args.epoch):

            self.model.train()

            pbar = tqdm(self.trainloader)
            pbar.set_description("Epoch {}/{}".format(e, self.args.epoch))

            total_loss = []

            for idx, (wav, landmark, reference) in enumerate(pbar):

                wav = wav.to(self.device)
                landmark = landmark.to(self.device)
                reference = reference.to(self.device)

                pred = self.model(wav, reference, use_ref=self.args.add_ref)

                # print(pred.shape)
                # print(landmark.shape)

                loss = self.Loss(pred, landmark)

                loss.backward()
                if self.args.PASE_optim:
                    self.pase_optim.step()
                
                self.mlp_optim.step()

                total_loss.append(loss.item())

                self.step += 1

                if (idx + 1) % self.args.log_freq == 0:
                    self._logger(loss, self.step, e, pbar, pred.cpu().detach(), landmark.cpu().detach())

            pbar.write("avg train loss: {}".format(np.mean(total_loss)))
            pbar.write("saving it to{}".format(self.args.save_path))
            if self.args.PASE_optim:
                self.pase_saver.save(self.pase_saver.prefix[:-1], e)

            self.decoder_saver.save(self.decoder_saver.prefix[:-1], e)

            self._eval(e)


    def _eval(self, epoch):

        self.model.eval()

        pbar = tqdm(self.validloader)
        pbar.set_description("EVAL {}/{}".format(epoch, self.args.epoch))
        total_loss = []
        with torch.no_grad():
            for idx, (wav, landmark, reference) in enumerate(pbar):
                self.model.eval()
                wav = wav.to(self.device)
                landmark = landmark.to(self.device)
                reference = reference.to(self.device)

                pred = self.model(wav, reference, use_ref=self.args.add_ref).detach()

                loss = self.RMSELoss(pred, landmark).detach()

                total_loss.append(loss.item())
        
        valid_loss = np.mean(total_loss)
        self._logger(valid_loss, -1, epoch, pbar)



    def _logger(self, loss, step, epoch, pbar, pred=None, label=None):

        if self.model.training:
            self.writer.add_scalar("train/Loss", loss.item(), global_step=step)
            self.writer.add_histogram("tain/pred", pred.data, global_step=step)
            self.writer.add_histogram("train/gt", label.data, global_step=step)
            if self.args.PASE_optim:
                pase_lr = self.pase_scheduler(self.pase_optim, step)
                mlp_lr = self.mlp_scheduler(self.mlp_optim, step)
                pbar.write("step: {}, loss: {}, PASE_lr: {}, Decoder_lr: {}".format(step, loss.item(), pase_lr, mlp_lr))
            else:
                mlp_lr = self.mlp_scheduler(self.mlp_optim, step)
                pbar.write("step: {}, loss: {}, Decoder_lr: {}".format(step, loss.item(), mlp_lr))


        else:
            self.writer.add_scalar("eval/Loss", loss, global_step=epoch)
            pbar.write("avg validation loss: {}".format(loss))
            if loss < self.best_loss:
                self.best_loss = loss
                # pbar.write("better model found! saving it to{}".format(self.args.save_path))

    def resume_training(self):

        # pase_state = self.pase_saver.read_latest_checkpoint()
        # decoder_state = self.decoder_saver.read_latest_checkpoint()
        # epoch = self.pase_saver.load_ckpt_step(pase_state)
        try:
            if self.args.PASE_optim:
                pase_state = self.pase_saver.read_latest_checkpoint()
                self.pase_saver.load_pretrained_ckpt(os.path.join(self.args.save_path,
                                                                        'weights_' + pase_state),
                                                        load_last=True)

            decoder_state = self.decoder_saver.read_latest_checkpoint()
            epoch = self.decoder_saver.load_ckpt_step(decoder_state)
            self.decoder_saver.load_pretrained_ckpt(os.path.join(self.args.save_path,
                                                                    'weights_' + decoder_state),
                                                    load_last=True)
        except TypeError:
            return 0

        return epoch + 1

    def find_close_lip(self, landmark):
        min_frame = 0
        min_sum = np.Infinity
        for batch in range(batch):
            for i in range(75):
                if np.sum(landmark[batch, i, 60:68]) < min_sum:
                    min_frame = i
                    min_sum = np.sum(landmark[batch, i, 60:68])

        reference_landmark = landmark[i,:]
        return reference_landmark

    def early_stopping(self):

        dataset = self.trainloader.dataset

        wav, landmark, ref = dataset.__getitem__(11)

        landmark_close = landmark[0].unsqueeze(0).to(self.device)
        landmark_open = landmark[48].unsqueeze(0).to(self.device)

        wav = wav.unsqueeze(0).repeat(2, 1, 1).to(self.device)

        landmark = torch.cat([landmark_open, landmark_close], dim=0)

        print("+" * 50)
        print("earling stopping...")

        loss = 10000
        step = 0
        while loss > 10:
                
            pred = self.model(wav, None, True)
            
            pred_close = pred[0, 0, :].view(1, 1, 136)
            pred_open = pred[0, 48, :].view(1, 1, 136)

            pred = torch.cat([pred_open, pred_close], dim=0)

            loss = self.Loss(pred, landmark)
            loss.backward()

            self.pase_optim.step()
            self.mlp_optim.step()
            step += 1

            # print(loss, step)

            if (step + 1) % 50 == 0:
                print("step: {}, loss: {}".format(step, loss.detach().cpu().item()))

    


        





        
            

        


        
        

