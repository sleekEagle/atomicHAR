import torch.nn as nn
import torch.nn.functional as F
from models.Encoder import CNN,Linear_encoder,Atom_encoder_CNN
from models.Transformer import TransformerModel
from models.Decoder import CNN_imu_decoder
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
        self.transformer=TransformerModel(self.tr_conf.d_model,
                                          self.tr_conf.n_head,
                                          self.tr_conf.dim_feedforward,
                                          self.tr_conf.num_layers,
                                          self.tr_conf.d_out,
                                          self.tr_conf.dropout)
        self.decoder=CNN_imu_decoder(1).double()


        
    def forward(self, x):
        #encooding
        atom_emb=self.cnn(x)
        atom_emb=torch.swapdims(atom_emb,0,1)
        trns_out=self.transformer(atom_emb)[-1,:,:]
        #reconstruct signal
        reconst=self.decoder(trns_out)

        return reconst
    
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