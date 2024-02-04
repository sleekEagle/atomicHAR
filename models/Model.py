import torch.nn as nn
import torch.nn.functional as F
from models.Encoder import CNN,Linear_encoder,Atom_encoder_CNN
from models.Transformer import TransformerModel
from models.Decoder import CNN_imu_decoder,CNN_atom_decoder,Activity_classifier_LIN
import torch
import utils
import matplotlib.pyplot as plt

class AtomicHAR(nn.Module):
    def __init__(self,conf,n_activities):
        super().__init__()
        self.cnn=CNN().double()
        print('in model. conf:')
        print(conf)
        self.tr_conf=conf.transformer
        self.thr=0.47256

        
    def forward(self, x):
        #encooding
        cnn_out,cnn_weights=self.cnn(x)
        atom_detected=cnn_out>self.thr
        atom_detected[:,:,0:2]=0
        atom_detected[:,:,-2:-1]=0

        plt.plot(atom_detected[0,0,:])

        return 1
    
    def get_max(self,t,window_size,step):
        windows=t.unfold(1,window_size,step)
        amax=torch.argmax(windows,dim=2)
        bs,winlen,nwin=windows.shape
        windows_=torch.zeros_like(windows)
        windows_=torch.reshape(windows_,(-1,nwin))
        amax=torch.reshape(amax,(-1,1))
        windows_[torch.arange(bs*winlen),amax[:,0].long()]=1
        windows_=torch.reshape(windows_,(bs,winlen,nwin))
        windows=torch.reshape(windows,(bs,winlen,nwin))
        windows=windows*windows_
        windows=torch.reshape(windows,(bs,-1))
        return windows