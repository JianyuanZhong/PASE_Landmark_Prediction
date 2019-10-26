import torch
import torch.nn.functional as F
import soundfile as sf
import os, sys
import json
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm


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

    device = get_freer_gpu()

    with open(args.data_cfg, "r") as cfg_file:
        data_cfg =json.load(cfg_file)

    trainset = data_cfg['train']['wav']
    wav_root = data_cfg['audio_root']
    # random sample |trainset| / 50 number of speaker to calcuate overal stat
    print("calcuating overal statistic by simple random simple...")
    
    signals = []
    sampled_speaker = []
    pbar = tqdm(trainset)
    
    for i, wav in enumerate(pbar):
        if i % 50 == 0:
            # read signal and downsample
            signal, frames = sf.read(os.path.join(wav_root, wav))
            signal = signal.astype(np.float32)
            signal = torch.from_numpy(signal).to(device).view(1, 1, signal.shape[-1])
            signal = F.interpolate(signal, size=48000, mode='linear', align_corners=True)
            signals.append(signal)
    
    signals = torch.cat(signals, dim=0)
    mean = torch.mean(torch.mean(signals, dim=2), dim=0).cpu()
    std = torch.std(torch.std(signals, dim=2), dim=0).cpu()

    speech_stat = {'mean' : mean, 'std' : std}
    print(speech_stat)

    print('done!')

    print("calculating overal statistic for landmarks...")

    landmark_cfg = data_cfg['train']['landmark']
    landmark_root = data_cfg['landmark_root']

    landmarks = []
    pbar = tqdm(landmark_cfg)
    for i, landmark in enumerate(pbar):
        landmarks.append(np.load(os.path.join(landmark_root, landmark)))

    landmarks = np.asarray(landmarks)
    landmarks = landmarks.reshape(landmarks.shape[0]*landmarks.shape[1],landmarks.shape[2],landmarks.shape[3])

    max_val = np.max(landmarks, 0)
    min_val = np.min(landmarks, 0)

    landmark_stat = {'max' : max_val, 'min' : min_val}

    print(landmark_stat)

    print("done!")

    stat = {
        "wav" : speech_stat,
        "landmark" : landmark_stat
    }

    output_dir = '/'.join(args.out_dir.split('/')[:-1])
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(args.out_dir, "wb") as out:
        pkl.dump(stat, out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_cfg", type=str, help="dir to data config file", default='/home/cxu-serve/u1/jzhong9/landmarkGeneration/data/oppo_all.cfg')
    parser.add_argument("--out_dir", type=str, help="dir to output statistic file", default='/home/cxu-serve/u1/jzhong9/landmarkGeneration/data/oppo_stat.pkl')
    args = parser.parse_args()

    print(args)

    main(args)