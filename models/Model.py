import torch.nn as nn
import torch.nn.functional as F
from models.Encoder import CNN,Linear_encoder
from models.Transformer import TransformerModel
from models.Decoder import Linear_decoder
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
        self.decoder=Linear_decoder().double()
        self.encoder=Linear_encoder().double()

    def forward(self, x):
        cnn_out=self.cnn(x)
        cnn_out=torch.swapaxes(cnn_out,0,2)
        cnn_out=torch.swapaxes(cnn_out,1,2)

        # lin_out=self.encoder(x)

        trans_out=self.transformer(cnn_out)
        embeddings=trans_out[39::40,:,:]
        embeddings=torch.swapaxes(embeddings,0,1)
        bs,seq,dim=embeddings.shape
        embeddings=embeddings.reshape(-1,dim)

        #decode to recreate the spacial signal
        gen=self.decoder(embeddings)
        return gen