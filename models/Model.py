import torch.nn as nn
import torch.nn.functional as F
from models.Encoder import CNN,Linear_encoder,Atom_encoder_CNN
from models.Transformer import TransformerModel
from models.Decoder import CNN_imu_decoder,CNN_atom_decoder
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
        print('in model. conf:')
        print(conf)
        self.tr_conf=conf.transformer
        '''
        transformer input: se_len, batch_size, emb_dim
        '''
        self.transformer=TransformerModel(d_model=self.tr_conf.d_model,
                                     n_head=self.tr_conf.n_head,
                                     dim_feedforward=self.tr_conf.dim_feedforward,
                                     num_layers=self.tr_conf.num_layers,
                                     d_out=self.tr_conf.d_out,
                                     dropout=self.tr_conf.dropout,
                                     device=0)
        self.imu_feat_dim=conf.cnn.imu_feat_dim
        self.imu_decorder_in_channels=int(self.imu_feat_dim/4)
        forcast_hidden=conf.forcast.hidden_dim

        self.transformer=self.transformer.double()
        self.imu_decoder=CNN_imu_decoder(self.imu_decorder_in_channels).double()
        self.atom_decoder=CNN_atom_decoder().double()
        self.atom_encoder=Atom_encoder_CNN().double()

        
        self.lin_bridge1 = nn.Linear(32*2, self.imu_feat_dim).double()
        self.lin_forcast1=nn.Linear(self.imu_feat_dim,forcast_hidden).double()
        self.lin_forcast2=nn.Linear(forcast_hidden,self.imu_feat_dim).double()
        self.lin_forcast3=nn.Linear(forcast_hidden,self.imu_feat_dim).double()

        self.encoder=Linear_encoder().double()
        self.leak=nn.LeakyReLU()

        #parameters to control the max segmentation
        self.thr=0.0001
        self.half_window=2
        self.imu_resample_len=20
        #maximum length of an atom in number of segments
        self.max_atom_len=10

        print('model parameters...')
        print((conf.cnn.imu_feat_dim,conf.forcast.hidden_dim))
        
    def forward(self, x,imu_mask,imu_len):
        #encooding
        bs,seq,dim,l=x.shape
        '''
        x.shape = [bs,seq,6,20]
        imu_input.shape = [bs*seq,6,20]
        imu_input:
        [ x[0,seq,6,20] , 
          x[1,seq,6,20] ,  
          x[2,seq,6,20] ] 
                        
        '''
        imu_input=torch.reshape(x,(-1,dim,l))
        cnn_out=self.cnn(imu_input)
        l,_,_=cnn_out.shape
        cnn_out=cnn_out.view(l,-1)
        bridge_out=F.sigmoid(self.lin_bridge1(cnn_out))
        '''
        shape of bridge_out:
        bd : dimention of bridge dimention
        [ [bd][bd]....  [bd][bd]....  [bd][bd]....  ]
            seq            seq          seq   ....bs number
        '''

        #**********forcasting the next sequence*********************************
        #forcasting
        forcast_in=torch.reshape(bridge_out,(bs,seq,-1))
        '''
        forcast_in.shape = [bs,seq,bd]
        [ [[bd][bd]...seq number] ,
          [[bd][bd]...seq number] ,
          [[bd][bd]...seq number] 
          ...bs number ]
        '''
        forcast_in_shft=F.pad(forcast_in,(0,0,1,0),"constant",0)
        forcast_in_shft=forcast_in_shft[:,0:-1,:]
        #forcasting mask. mask the first and the last items in the sequence
        #and mask as given in the imu_mask
        bs,seq,d=forcast_in.shape
        forcast_mask=torch.ones_like(forcast_in)
        forcast_mask[:,0,:]=0
        forcast_mask=torch.reshape(forcast_mask,(-1,d))
        imu_mask=imu_mask[:,:,0,0].unsqueeze(2)
        imu_mask_=imu_mask.repeat(1,1,d)
        imu_mask_=torch.reshape(imu_mask_,(-1,d))
        forcast_mask=forcast_mask*imu_mask_
        
        forcast_in=torch.reshape(forcast_in,(-1,d))
        '''
        forcast_in.shape = [bs*seq, bd]
        [[bd][bd]...seq number [bd][bd]...seq number [bd][bd]...seq number]
        [bd][bd]...seq number times bs
        '''
        forcast_in_shft=torch.reshape(forcast_in_shft,(-1,d))

        #forcast the next step and calculate the loss
        #**********read!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #check is the forcast loss goes down to 0.00004 level
        #change the number of layers. this this is achieved consistantly. 
        #now the convergence is pretty noisy.

        forcast_feat=F.relu(self.lin_forcast1(forcast_in_shft))
        forcast=self.lin_forcast2(forcast_feat)

        forcast_loss=torch.mean(torch.square(forcast*forcast_mask-forcast_in*forcast_mask),dim=1)
        #select forcast loss threshold dynamically
        # sorted, indices=torch.sort(forcast_loss)
        # l=int(sorted.shape[0]*0.9)
        # forcast_loss_thr=sorted[l].item()
        forcast_loss=torch.reshape(forcast_loss,(bs,seq))
        '''
        forcast_loss.shape=[bs,seq]
        [loss, loss, loss, loss ......seq number,
        loss, loss, loss, loss ......seq number,
        loss, loss, loss, loss ......seq number,
        ......bs number]
        '''
        
        #************************************************************************

        #find segment points using forcast loss**********************************
        # loss_mask=torch.ones_like(forcast_loss)
        # loss_mask[:,0:2]=0
        # loss_mask[:,-2:]=0
        # forcast_valid=forcast_loss>forcast_loss_thr
        # forcast_valid=forcast_valid*loss_mask
        # forcast_loss_=forcast_loss*forcast_valid
        # forcast_loss_selected=self.get_max(forcast_loss_,self.half_window*2,self.half_window*2)
        # forcast_loss_second=forcast_loss_selected[:,self.half_window:]
        # forcast_loss_selected_=self.get_max(forcast_loss_second,self.half_window*2,self.half_window*2)
        # forcast_loss_selected_=F.pad(forcast_loss_selected_,(self.half_window,0),"constant", 0)
        # bs,l=forcast_loss_selected.shape
        # _,l_=forcast_loss_selected_.shape
        # additional_padding=l-l_
        # if additional_padding>0:
        #     forcast_loss_selected_=F.pad(forcast_loss_selected_,(0,self.half_window),"constant", 0)
        # seg_points=forcast_loss_selected_>0
        # seg_point_args=torch.where(seg_points)

        #obtain segmentation points per each batch
        # imu_last_seg=torch.round(imu_len/seq)
        # seg_len_list=[]
        # for b in range(bs):
        #     b_arg=torch.argwhere(seg_point_args[0]==b)[:,0]
        #     b_seg_points=seg_point_args[1][b_arg]
        #     b_seg_points=torch.cat((torch.zeros(1),
        #                             b_seg_points,
        #                             torch.unsqueeze(imu_last_seg[b],dim=0)),dim=0)
        #     seg_lens=torch.diff(b_seg_points).numpy()
        #     seg_lens=list(seg_lens)
        #     seg_lens=[int(item) for item in seg_lens]
        #     #combine nearby segments if one of them is too short
        #     seg_lens_mod=[]
        #     i=0
        #     while i<len(seg_lens):
        #         len_sum=seg_lens[i]
        #         for j in range(i+1,len(seg_lens)):
        #             if seg_lens[j]>1:
        #                 break
        #             else:
        #                 i+=1
        #                 len_sum+=seg_lens[j]
        #         i+=1
        #         seg_lens_mod.append(len_sum)
        #     seg_len_list.append(seg_lens_mod)
        #***************************************************************************
                

        #*******transformer*********************************************************
        #create the 20 x 20 transformer mask here
        # bs,l=seg_points.shape
        # mask=torch.empty(0)
        # for b in range(bs):
        #     mask_=torch.full((1,l,l),float('-inf'),dtype=torch.float64)
        #     ind_intervals=seg_len_list[b]
        #     last_idx=0
        #     for idx in ind_intervals:
        #         mask_[0,last_idx:(last_idx+idx),last_idx:(last_idx+idx)]=0
        #         last_idx+=idx
        #     mask_=mask_.repeat(self.tr_conf.n_head,1,1)
        #     mask=torch.cat((mask,mask_),dim=0)
        # mask_.repeat(self.tr_conf.n_head,1,1)
        # mask=mask.double()

        # bridge_out_tr=torch.reshape(bridge_out,(bs,seq,-1))
        '''
        bridge_out_tr.shape = [bs, seq, bd]
        [ [bd][bd][bd][bd][bd].....seq number ,
          [bd][bd][bd][bd][bd].....seq number ,
          [bd][bd][bd][bd][bd].....seq number ,
          [bd][bd][bd][bd][bd].....seq number ,
          ........ bs number]
        '''
        # seg_featurs=torch.empty(0)
        # for b in range(bs): 
        #     b_seg_points=torch.cumsum(torch.tensor(seg_len_list[b]),dim=0)
        #     b_seg_points=list(b_seg_points.numpy())
        #     last_index=0
        #     for sp in b_seg_points:
        #         padding=self.max_atom_len-(sp-last_index)
        #         feat=bridge_out_tr[b,last_index:sp,:]
        #         feat=F.pad(feat,(0,0,padding,0),"constant", 0)
        #         feat=torch.unsqueeze(feat,dim=0)
        #         seg_featurs=torch.cat((seg_featurs,feat),dim=0)

        # seg_featurs=torch.swapaxes(seg_featurs,1,2)
        # atom_emb=self.atom_encoder(seg_featurs)

        # l,_=bridge_out.shape
        bridge_out=bridge_out.view(l,self.imu_decorder_in_channels,4)

        # tr_out=self.transformer(tr_input)      

        #IMU decoder: segment level*******************************************************************
        imu_gen=self.imu_decoder(bridge_out)
        _,dim,l=imu_gen.shape
        imu_gen=torch.reshape(imu_gen,(bs,seq,dim,l))
        #**********************************************************************************************

        #IMU decoder: atomic level*********************************************************************
        # n_seg,_=atom_emb.shape
        # atom_decoder_in=torch.reshape(atom_emb,(n_seg,4,4))
        # atom_gen=self.atom_decoder(atom_decoder_in)
        #**********************************************************************************************

        output={}
        output['imu_gen']=imu_gen
        # output['atom_gen']=atom_gen
        # output['imu_last_seg']=imu_last_seg
        # output['seg_len_list']=seg_len_list
        output['bridge_out']=bridge_out
        output['forcast_real']=forcast_in
        output['forcast']=forcast
        output['forcast_mask']=forcast_mask
        output['forcast_loss']=forcast_loss
        #**********************************************************************************

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