import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import json
import pickle as pkl
import argparse
import os, sys
import numpy as np
import cv2 as cv
from core.model.pase2landmark import pase2landmark, paseLSTMLandmark
from core.dataset import audio2landmark, audio2landmark_norm
import subprocess


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

def vis(lists, img_path, out_dir): # a list [frame, landmark, frame, landmark]
        windows = len(lists)
        fig = plt.figure(figsize=plt.figaspect(.5))
        print(lists.shape, windows)
        frames = []
        for i in range(windows):
            frame_id = i + 1
            print(os.path.join(img_path, "0000{}.png".format(frame_id)))
#             fig = plt.figure(figsize=plt.figaspect(.5))
            if frame_id < 10:
                img = cv.imread(os.path.join(img_path, "0000{}.png".format(frame_id)))
            else:
                img = cv.imread(os.path.join(img_path, "000{}.png".format(frame_id)))
            img = cv.resize(img ,None,fx=0.5714,fy=0.5714)
            preds = lists[i]
            ax = plt
#             ax = fig
            ax.imshow(img)
            ax.plot(preds[0:17,0],preds[0:17,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
            ax.plot(preds[17:22,0],preds[17:22,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
            ax.plot(preds[22:27,0],preds[22:27,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
            ax.plot(preds[27:31,0],preds[27:31,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
            ax.plot(preds[31:36,0],preds[31:36,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
            ax.plot(preds[36:42,0],preds[36:42,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
            ax.plot(preds[42:48,0],preds[42:48,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
            ax.plot(preds[48:60,0],preds[48:60,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
            ax.plot(preds[60:68,0],preds[60:68,1],marker='o',markersize=1,linestyle='-',color='b',lw=1)
            ax.axis('off')
            
            plt.savefig(out_dir + "/%03d.png" % i)
            plt.close()

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


    test_cfg = data_cfg['test']
    audio_root = data_cfg['audio_root']
    landmark_root = data_cfg['landmark_root']

    device = get_freer_gpu()
    print('=' * 50)
    print('Using device: {}'.format(device))
    print('=' * 50)

    if args.model == 'MLP':
        model = pase2landmark(PASE_cfg, MLP_cfg, args.PASE_ckpt).to(device)
    elif args.model == 'LSTM':
        model = paseLSTMLandmark(PASE_cfg, MLP_cfg, args.PASE_ckpt).to(device)

    model.decoder.load_pretrained(args.model_ckpt, load_last=True)

    testset = audio2landmark_norm(test_cfg, audio_root, landmark_root, stat, norm=False)

    wav, landmark, ref = testset.__getitem__(1)
    wav, landmark, ref = wav.unsqueeze(0).to(device), landmark.unsqueeze(0).to(device), ref.unsqueeze(0).to(device)
    model.eval()
    pred = model(wav, ref, use_ref=True)

    pred = pred.detach().cpu().view(pred.shape[0], pred.shape[1],pred.shape[2]//2, 2).squeeze()
    pred = pred.numpy()
    landmark = landmark.detach().cpu().view(landmark.shape[0], landmark.shape[1],landmark.shape[2]//2, 2).squeeze()
    landmark = landmark.numpy()

    wav_file = test_cfg['wav'][1]
    img_path = os.path.join("/home/cxu-serve/p1/jzhong9/data/oppo/img/", wav_file.split('.')[0])

    vis(pred, img_path, os.path.join(args.out_dir, "pred"))
    # vis(landmark, img_path, os.path.join(args.out_dir, "gt"))

    if os.path.exists(os.path.join(args.out_dir, "pred/pred.mp4")):
        os.remove(os.path.join(args.out_dir, "pred/pred.mp4"))
    subprocess.call(["ffmpeg", "-framerate", "25", "-i", os.path.join(args.out_dir, "pred/%03d.png"), os.path.join(args.out_dir, "pred/pred.mp4")])
    subprocess.call(["ffmpeg", "-i", os.path.join(args.out_dir, "pred/pred.mp4"), "-i", os.path.join(audio_root, wav_file),  "-map", "0:v", "-map", "1:a", "-c", "copy", "-shortest", os.path.join(args.out_dir, "pred/output.mkv")])

    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str, help="dir to data config file", default='/home/cxu-serve/u1/jzhong9/landmarkGeneration/data/oppo_all.cfg')
    parser.add_argument("--model", type=str, default="MLP")
    parser.add_argument("--stat", type=str, help="dir to output statistic file", default='/home/cxu-serve/u1/jzhong9/landmarkGeneration/data/oppo_stat.pkl')
    parser.add_argument("--PASE_cfg", type=str, help="path to pase config file", default="/home/cxu-serve/u1/jzhong9/pase/cfg/PASE_MAT_sinc_jiany_512.cfg")
    parser.add_argument("--MLP_cfg", type=str, help="path to mlp config file", default="cfg/MLP.cfg")
    parser.add_argument("--PASE_ckpt", type=str, help="path to pase ckpt")
    parser.add_argument("--model_ckpt", type=str)
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    print(args)

    main(args)