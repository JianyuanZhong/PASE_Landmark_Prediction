import torch
import torch.nn as nn
import torch.nn.functional as F
from pase.models.modules import Model
from pase.models.frontend import wf_builder
from core.model.neural_networks import MLP, LSTM_cudnn



class pase2landmark(Model):

    def __init__(self, PASE_cfg, MLP_cfg, PASE_ckpt, context_left=0, context_right=0):
        super().__init__()
        input_dim = 4 * PASE_cfg['rnn_dim'] * (1 + context_left + context_right)
        self.context_left = context_left
        self.context_right = context_right
        
        self.pase = wf_builder(PASE_cfg)
        self.pase.load_pretrained(PASE_ckpt, load_last=True, verbose=False)
        self.decoder = MLP(MLP_cfg, input_dim)



    def forward(self, wav, reference=None, early_stop=False, use_ref=False, movement=25, temp=1.15):
        # wav = F.interpolate(wav, size=48000, mode='linear', align_corners=True)

        # bs x 1 x 48000 -> bs x 512 x 300
        if not early_stop:
            wav = self.pase(wav)
        else:
            with torch.no_grad():
                wav = self.pase(wav)
        
        # bs x 512 x 300 -> bs x 300 x 512
        wav = wav.permute(0, 2, 1).contiguous()

        # bs x 300 x 512 -> bs x 75 x 4 x 512
        wav = wav.view(wav.shape[0], wav.shape[1] // 4, 4, wav.shape[2])

        landmarks = []
        for i in range(wav.shape[1]):

            begin = max(0, i-self.context_left)

            end = min(i+self.context_right+1, wav.shape[1])
            
            # bs x (left + 1 + right) x 4 x 512 
            x = wav[:, begin:end, :, :].contiguous()

            # padding with zero tensor
            if i-self.context_left < 0:
                x = torch.cat([torch.zeros(x.shape[0], self.context_left-i, x.shape[2], x.shape[3]).to(x.get_device()), x], dim=1)


            if i+self.context_right+1 > wav.shape[1]:
                x = torch.cat([x, torch.zeros(x.shape[0], i+self.context_right+1-wav.shape[1], x.shape[2], x.shape[3]).to(x.get_device())], dim=1)


            # bs x (left + 1 + right) x 512 -> bs x ((left + 1 + right) * 4 * 512)
            x = x.view(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])


            # bs x ((left + 1 + right) * 4 * 512) -> bs x 138
            x = self.decoder(x).unsqueeze(1)

            if use_ref:
                
                x = torch.tanh(x)

                alpha = F.softmax(x, dim=2)
                
                x_movement = alpha * movement

                x = x_movement * x

            # x = F.tanh(x)
            
            landmarks.append(x)
        # print(reference.shape)
        
        alpha = 0.8

        landmarks = torch.cat(landmarks, dim=1) 
        if use_ref:
            landmarks = landmarks + reference.unsqueeze(1).repeat(1, landmarks.shape[1], 1)
            
        return landmarks


class paseLSTMLandmark(Model):

    def __init__(self, PASE_cfg, LSTM_cfg, PASE_ckpt):
        super().__init__()

        self.pase = wf_builder(PASE_cfg)
        self.pase.load_pretrained(PASE_ckpt, load_last=True, verbose=False)

        input_dim = 4 * PASE_cfg['rnn_dim']
        
        self.decoder = nn.ModuleList([LSTM_cudnn(LSTM_cfg, input_dim),
                                        nn.Linear(136, 136)])


        
        

    def forward(self, wav, reference, use_ref=False, movement=2, temp=0.8):
        
        wav = self.pase(wav)

        # bs x 512 x 300 -> bs x 300 x 512
        wav = wav.permute(0, 2, 1).contiguous()

        # bs x 300 x 512 -> bs x 75 x 4 x 512
        wav = wav.view(wav.shape[0], wav.shape[1] // 4, 4 * wav.shape[2])

        x = self.decoder[0](wav)

        if use_ref:
                
                x = torch.tanh(x)

                alpha = F.softmax(x, dim=2) * temp
                
                x_movement = alpha #* movement

                x = x_movement * x

                pred = self.decoder[1](x) + reference.unsqueeze(1).repeat(1, x.shape[1], 1)


        return pred

class featureLSTMLandmark(Model):

    def __init__(self, LSTM_cfg, input_dim):
        super().__init__()
        self.decoder = LSTM_cudnn(LSTM_cfg, 4 * input_dim)

    def forward(self, feature, reference, use_ref, movement=2, temp=0.8):

        x = feature.permute(0, 2, 1).contiguous()

        x = x.view(x.shape[0], x.shape[1] // 4, 4 * x.shape[2])

        x = self.decoder(x)

        if use_ref:
                
            x = torch.tanh(x)

            alpha = F.softmax(x, dim=2) * temp
                
            x_movement = alpha * movement

            x = x_movement * x

            pred = x + reference.unsqueeze(1).repeat(1, x.shape[1], 1)

        return pred





