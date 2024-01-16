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

        self.lin_bridge1 = nn.Linear(32*2, 4).double()
        self.lin_bridge2 = nn.Linear(32, 32).double()
        self.lin_forcast=nn.Linear(4,4).double()

        self.encoder=Linear_encoder().double()
        self.leak=nn.LeakyReLU()

        #parameters to control the max segmentation
        self.thr=0.132
        self.half_window=2

    def forward(self, x):
        #encooding
        bs,seq,dim,l=x.shape
        imu_input=torch.reshape(x,(-1,dim,l))
        cnn_out=self.cnn(imu_input)
        l,_,_=cnn_out.shape
        cnn_out=cnn_out.view(l,-1)
        bridge_out=F.sigmoid(self.lin_bridge1(cnn_out))

        forcast_in=torch.reshape(bridge_out,(bs,seq,-1))
        forcast_in_shft=F.pad(forcast_in,(0,0,1,0),"constant",0)
        forcast_in_shft=forcast_in_shft[:,0:-1,:]
        forcast_mask=torch.ones_like(forcast_in)
        forcast_mask[:,0,:]=0
        _,_,d=forcast_in.shape
        forcast_in=torch.reshape(forcast_in,(-1,d))
        forcast_in_shft=torch.reshape(forcast_in_shft,(-1,d))
        forcast_mask=torch.reshape(forcast_mask,(-1,d))

        #forcast the next step
        forcast=self.lin_forcast(forcast_in_shft)
        forcast=forcast*forcast_mask
        forcast_loss=torch.mean(torch.square(forcast-forcast_in),dim=1)
        forcast_loss=torch.reshape(forcast_loss,(bs,seq))

        #find segment points
        loss_mask=torch.ones_like(forcast_loss)
        loss_mask[:,0:2]=0
        loss_mask[:,-2:]=0
        forcast_valid=forcast_loss>0
        forcast_valid=forcast_valid*loss_mask
        forcast_loss=forcast_loss*forcast_valid
        forcast_loss_selected=self.get_max(forcast_loss,self.half_window*2,self.half_window*2)
        forcast_loss_second=forcast_loss_selected[:,self.half_window:]
        forcast_loss_selected_=self.get_max(forcast_loss_second,self.half_window*2,self.half_window*2)
        forcast_loss_selected_=F.pad(forcast_loss_selected_,(self.half_window,0),"constant", 0)
        bs,l=forcast_loss_selected.shape
        _,l_=forcast_loss_selected_.shape
        additional_padding=l-l_
        if additional_padding>0:
            forcast_loss_selected_=F.pad(forcast_loss_selected_,(0,self.half_window),"constant", 0)


        # bridge_out=torch.round(bridge_out,decimals=10)
        # bridge_out=F.relu(self.lin_bridge2(bridge_out))
        
        #transformer
        #create the 20 x 20 transformer mask here
        tr_input=torch.reshape(bridge_out,(seq,bs,-1))
        l,_=bridge_out.shape
        bridge_out=bridge_out.view(l,2,2)


        # tr_out=self.transformer(tr_input)
        # _,_,dim=tr_out.shape
        # tr_out=torch.reshape(tr_out,(-1,dim))
        # tr_out=tr_out.view(seq*bs,8,4)

        #xyz decoder
        # xyz_gen=self.xyz_decoder(tr_out)
        # _,dim,l=xyz_gen.shape
        # xyz_gen=torch.reshape(xyz_gen,(bs,seq,dim,l))

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

        output={}
        output['imu_gen']=imu_gen
        output['bridge_out']=bridge_out
        output['forcast_real']=forcast_in
        output['forcast']=forcast
        output['forcast_mask']=forcast_mask
        output['forcast_loss']=forcast_loss

        return output
    
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