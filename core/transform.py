import librosa

class MFCC_librosa(object):

    def __init__(self, n_fft=2048, hop=160,
                 order=13, sr=16000, win=400,der_order=2,n_mels=40,
                 htk=True, name='mfcc_librosa'):
        self.hop = hop
        # Santi: the librosa mfcc api does not always
        # accept a window argument, so we enforce n_fft
        # to be window to ensure the window len restriction
        #self.win = win
        self.n_fft = win
        self.order = order
        self.sr = 16000
        self.der_order=der_order
        self.n_mels=n_mels
        self.htk=True
        self.name = name

    # @profile
    def __call__(self, wav):
        y = wav
        max_frames = y.shape[0] // self.hop
        mfcc = librosa.feature.mfcc(y, sr=self.sr,
                                        n_mfcc=self.order,
                                        n_fft=self.n_fft,
                                        hop_length=self.hop,
                                        #win_length=self.win,
					                    n_mels=self.n_mels,
                                        htk=self.htk,
                                        )[:, :max_frames]
        if self.der_order > 0 :
            deltas=[mfcc]
            for n in range(1,self.der_order+1):
                deltas.append(librosa.feature.delta(mfcc,order=n))
            mfcc=np.concatenate(deltas)

        pkg[self.name] = torch.tensor(mfcc.astype(np.float32))  
        return pkg

    def __repr__(self):
        attrs = '(order={}, sr={})'.format(self.order,
                                           self.sr)
        return self.__class__.__name__ + attrs