import torch.nn as nn
import torch.nn.functional as F
from models.CNN import CNN
from models.Transformer import TransformerModel
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

    def forward(self, x):
        cnn_out=self.cnn(x)
        cnn_out=torch.swapaxes(cnn_out,0,2)
        cnn_out=torch.swapaxes(cnn_out,1,2)
        trans_out=self.transformer(cnn_out)
        return trans_out