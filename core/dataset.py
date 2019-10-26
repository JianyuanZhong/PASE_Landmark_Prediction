import librosa
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset
import soundfile as sf
import os
import numpy as np
import pickle
import json
from pase.transforms import MFCC_librosa, FBanks, Gammatone, LPS, Prosody

class audio2landmark(Dataset):

    def __init__(self, data_cfg, audio_root, landmark_root, stat,device='cpu'):
        super(audio2landmark).__init__()
        self.wav_list = data_cfg['wav']
        self.landmark_list = data_cfg['landmark']
        self.stat = stat
        
        self.audio_root = audio_root
        self.landmark_root = landmark_root

        self.sample_rate = 48000

        self.device = device

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, index):
        wav_file  = os.path.join(self.audio_root, self.wav_list[index])
        landmark_file = os.path.join(self.landmark_root, self.landmark_list[index])

        if self.wav_list[index].split('.')[0] != self.landmark_list[index].split('_')[0]:
            raise ValueError("wav and landmark not aligned!! wav: {}, landmark:{}".format(wav_file, landmark_file))
        
        wav, rate = sf.read(wav_file)
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav[:,0]
        wav = torch.from_numpy(wav)
        wav = F.interpolate(wav.view(1, 1, wav.shape[-1]), size=48000, mode='linear', align_corners=True)
        wav = wav.squeeze(0)
        wav = (wav - self.stat['mean']) / self.stat['std']

        landmark = torch.from_numpy(np.load(landmark_file)).float()

        reference_landmark = landmark[0,:]

        return wav, landmark, reference_landmark

class audio2landmark_norm(Dataset):

    def __init__(self, data_cfg, audio_root, landmark_root, stat, feature='wav',norm=True, device='cpu'):
        super(audio2landmark).__init__()
        self.wav_list = data_cfg['wav']
        self.landmark_list = data_cfg['landmark']

        
        self.wav_stat = stat['wav']
        self.landmark_stat = stat['landmark']
        
        self.audio_root = audio_root
        self.landmark_root = landmark_root

        self.sample_rate = 48000
        self.feature = feature
        self.norm = norm

        self.device = device

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, index):
        wav_file  = os.path.join(self.audio_root, self.wav_list[index])
        landmark_file = os.path.join(self.landmark_root, self.landmark_list[index])

        if self.wav_list[index].split('.')[0] != self.landmark_list[index].split('_')[0]:
            raise ValueError("wav and landmark not aligned!! wav: {}, landmark:{}".format(wav_file, landmark_file))
        
        wav, rate = sf.read(wav_file)
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav[:,0]
        wav = torch.from_numpy(wav)
        wav = F.interpolate(wav.view(1, 1, wav.shape[-1]), size=48000, mode='linear', align_corners=True)
        wav = wav.squeeze(0)
        wav = (wav - self.wav_stat['mean']) / self.wav_stat['std']
        
        feature = self.transform(wav, self.feature)

        if self.norm:
            landmark = (np.load(landmark_file) - self.landmark_stat['min']) / (self.landmark_stat['max'] - self.landmark_stat['min'])
        else:
            landmark = np.load(landmark_file)

        landmark = torch.from_numpy(landmark).float().view(landmark.shape[0], landmark.shape[1] * landmark.shape[2])

        reference_landmark = landmark[0,:]

        return feature, landmark, reference_landmark

    def transform(self, wav, feature):
        pkg = {'chunk':wav.squeeze(), 'chunk_beg_i':0, 'chunk_end_i':wav.shape[1]}
        if feature == "MFCC":
            trans = MFCC_librosa()
            pkg = trans(pkg)
            return pkg['mfcc_librosa']
        elif feature == 'FBanks':
            trans = FBanks()
            pkg = trans(pkg)
            return pkg['fbank']
        elif feature == 'Gammatone':
            trans = Gammatone()
            pkg = trans(pkg)
            return pkg['gtn']
        elif feature == 'LPS':
            trans = LPS()
            pkg = trans(pkg)
            return pkg['lps']
        elif feature == 'Prosody':
            trans = Prosody()
            pkg = trans(pkg)
            return pkg['prosody']
        else:
            return wav

class audio2landmark_img(Dataset):

    def __init__(self, data_cfg, audio_root, landmark_root, stat,device='cpu'):
        super(audio2landmark).__init__()
        self.wav_list = data_cfg['wav']
        self.landmark_list = data_cfg['landmark']

        
        self.wav_stat = stat['wav']
        self.landmark_stat = stat['landmark']
        
        self.audio_root = audio_root
        self.landmark_root = landmark_root

        self.sample_rate = 48000

        self.device = device

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, index):
        wav_file  = os.path.join(self.audio_root, self.wav_list[index])
        landmark_file = os.path.join(self.landmark_root, self.landmark_list[index])

        if self.wav_list[index].split('.')[0] != self.landmark_list[index].split('_')[0]:
            raise ValueError("wav and landmark not aligned!! wav: {}, landmark:{}".format(wav_file, landmark_file))
        
        wav, rate = sf.read(wav_file)
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav[:,0]
        wav = torch.from_numpy(wav)
        wav = F.interpolate(wav.view(1, 1, wav.shape[-1]), size=48000, mode='linear', align_corners=True)
        wav = wav.squeeze(0)
        wav = (wav - self.wav_stat['mean']) / self.wav_stat['std']

        landmark = np.load(landmark_file)

        img = np.zeros((225, 255), np.float)

        for i in range(landmark.shape[0]):
            img[landmark[i, 0], landmark[i, 1]] = 1.0

        landmark = img

        landmark = torch.from_numpy(landmark).float().unsqueeze(0)

        reference_landmark = landmark[0,:]

        return wav, landmark, reference_landmark

    