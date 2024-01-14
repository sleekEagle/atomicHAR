import torch.nn as nn
import torch.nn.functional as F
from models.Encoder import CNN,Linear_encoder
from models.Transformer import TransformerModel
from models.Decoder import CNN_imu_decoder,CNN_xyz_decoder
import torch

class AtomicHAR(nn.Module):
    def __init__(self,conf):
        super().__init__()
        self.cnn=CNN(conf.cnn.in_channels,
                     conf.transformer.d_model,
                     input_size=400,
                     dilation=1,
                     kernel_size=3,
                     stride=1).double()
        tr_conf=conf.transformer
        '''
        transformer input: se_len, batch_size, emb_dim
        '''
        self.transformer=TransformerModel(d_model=tr_conf.d_model,
                                     n_head=tr_conf.n_head,
                                     dim_feedforward=tr_conf.dim_feedforward,
                                     num_layers=tr_conf.num_layers,
                                     d_out=tr_conf.d_out,
                                     dropout=tr_conf.dropout,
                                     device=0)
        self.transformer=self.transformer.double()
        self.imu_decoder=CNN_imu_decoder().double()
        self.xyz_decoder=CNN_xyz_decoder().double()

        self.lin_bridge1 = nn.Linear(32*2, 32).double()
        self.lin_bridge2 = nn.Linear(32, 32).double()

        self.encoder=Linear_encoder().double()

    def forward(self, x):
        #encooding
        bs,seq,dim,l=x.shape
        imu_input=torch.reshape(x,(-1,dim,l))
        cnn_out=self.cnn(imu_input)
        l,_,_=cnn_out.shape
        cnn_out=cnn_out.view(l,-1)
        bridge_out=F.relu(self.lin_bridge1(cnn_out))
        bridge_out=F.relu(self.lin_bridge2(bridge_out))
        
        #transformer
        tr_input=torch.reshape(bridge_out,(seq,bs,-1))
        bridge_out=bridge_out.view(l,8,4)

        tr_out=self.transformer(tr_input)
        _,_,dim=tr_out.shape
        tr_out=torch.reshape(tr_out,(-1,dim))
        tr_out=tr_out.view(seq*bs,8,4)

        #xyz decoder
        xyz_gen=self.xyz_decoder(tr_out)
        _,dim,l=xyz_gen.shape
        xyz_gen=torch.reshape(xyz_gen,(bs,seq,dim,l))

        # cnn_out=torch.swapaxes(cnn_out,0,2)
        # cnn_out=torch.swapaxes(cnn_out,1,2)

        # lin_out=self.encoder(x)

        # trans_out=self.transformer(cnn_out)
        # embeddings=trans_out[19::20,:,:]
        # embeddings=torch.swapaxes(embeddings,0,1)
        # bs,seq,dim=embeddings.shape
        # embeddings=embeddings.reshape(-1,dim)

        #IMU decoder
        imu_gen=self.imu_decoder(bridge_out)
        _,dim,l=imu_gen.shape
        imu_gen=torch.reshape(imu_gen,(bs,seq,dim,l))
        return imu_gen,xyz_gen