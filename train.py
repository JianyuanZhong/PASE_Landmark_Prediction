import torch
import torch.optim as optim
import json
import pickle as pkl
import argparse
import os, sys
import numpy as np
from tqdm import tqdm, trange
from core.dataset import audio2landmark, audio2landmark_norm, audio2landmark_img
from torch.utils.data import DataLoader
from core.trainer import Trainer
import warnings
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def get_freer_gpu(trials=10):
    for j in range(trials):
         os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
         memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
         dev_ = torch.device('cuda:'+str(np.argmax(memory_available)))
         try:
            a = torch.rand(1).cuda(dev_)
            return dev_
         except: 
            pass
            print('NO GPU AVAILABLE!!!')
            exit(1)

def main(args):
    with open(args.data_cfg, "r") as cfg:
        data_cfg = json.load(cfg)

    with open(args.PASE_cfg, "r") as PASE_cfg:
        print("=" * 50)
        PASE_cfg = json.load(PASE_cfg)
        print("PASE config: {}".format(PASE_cfg))

    with open(args.MLP_cfg, "r") as MLP_cfg:
        print("=" * 50)
        MLP_cfg = json.load(MLP_cfg)
        print("MLP config: {}".format(MLP_cfg))

    with open(args.stat, "rb") as stt_file:
        stat = pkl.load(stt_file)

    args.PASE_optim = str2bool(args.PASE_optim)
    args.save_best = str2bool(args.save_best)
    args.landmark_norm = str2bool(args.landmark_norm)
    args.early_stopping = str2bool(args.early_stopping)
    args.add_ref= str2bool(args.add_ref)
    print("=" * 50)
    print("Normalize landmark: {}".format(args.landmark_norm))

    print("=" * 50)
    print("Add Reference Landmark for trainning".format(args.add_ref))


    train_cfg = data_cfg['train']
    valid_cfg = data_cfg['dev']
    audio_root = data_cfg['audio_root']
    landmark_root = data_cfg['landmark_root']

    device = "cuda:0"#get_freer_gpu()
    print('=' * 50)
    print('Using device: {}'.format(device))
    print('=' * 50)

    if 'landmark' not in stat.keys():
        trainset = audio2landmark(train_cfg, audio_root, landmark_root, stat, device)
        validset = audio2landmark(valid_cfg, audio_root, landmark_root, stat, device)
    else:
        trainset = audio2landmark_norm(train_cfg, audio_root, landmark_root, stat, args.feature, args.landmark_norm, device)
        validset = audio2landmark_norm(valid_cfg, audio_root, landmark_root, stat, args.feature, args.landmark_norm, device)

    train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=True, drop_last=True)

    trainer = Trainer(PASE_cfg, MLP_cfg, train_loader, valid_loader, device, args)
    trainer.train()
    # pbar = tqdm(train_loader)
    # for i, (wav, landmark, ref) in enumerate(pbar):
    #     pbar.write(str(wav.shape))
    #     pbar.write(str(landmark.shape))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str, help="dir to data config file", default='/home/cxu-serve/u1/jzhong9/landmarkGeneration/data/oppo_all.cfg')
    parser.add_argument("--model", type=str, default="MLP")
    parser.add_argument("--stat", type=str, help="dir to output statistic file", default='/home/cxu-serve/u1/jzhong9/landmarkGeneration/data/oppo_stat.pkl')
    parser.add_argument("--epoch", type=int, default=50, help="num of epoches")
    parser.add_argument("--batch_size", type=int, default=28)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--PASE_cfg", type=str, help="path to pase config file", default="/home/cxu-serve/u1/jzhong9/pase/cfg/PASE_MAT_sinc_jiany_512.cfg")
    parser.add_argument("--MLP_cfg", type=str, help="path to mlp config file", default="cfg/MLP.cfg")
    parser.add_argument("--context_left", type=int, default=1, help="left context window for MPL")
    parser.add_argument("--context_right", type=int, default=1, help="right context window for MPL")
    parser.add_argument("--PASE_optim", type=str, default=None, help="use seperate optimizator for PASE encoder")
    parser.add_argument("--PASE_lr", type=float)
    parser.add_argument("--PASE_ckpt", type=str, help="path to pase ckpt")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--save_path", type=str, help="path to save ckpt")
    parser.add_argument("--save_best", type=str, default="1", help="save the best model")
    parser.add_argument("--log_freq", type=int, default=50, help="tensorboard log frequency")
    parser.add_argument("--landmark_norm", type=str, default="False", help="norm_for_landmark")
    parser.add_argument("--early_stopping", type=str, default="False")
    parser.add_argument("--add_ref", type=str, default="False")
    parser.add_argument("--feature", type=str, default='wav')
    args = parser.parse_args()

    print(args)

    main(args)

