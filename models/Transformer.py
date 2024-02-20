import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
import torch.nn.functional as F

class TransformerModel(nn.Module):

    def __init__(self, d_model: int, n_head: int, dim_feedforward : int,
                 num_layers: int, d_out:int, dropout: float = 0.5, device: int=0):
        super().__init__()
        self.device=device
        # self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, n_head, dim_feedforward , dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers).double()
        self.embedding = nn.Embedding(512, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model,d_out).double()


        # self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        # self.embedding.weight.data.uniform_(-initrange, initrange)
        # self.linear.bias.data.zero_()
        # self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[seq_len, batch_size]``
            src_mask: Tensor, shape ``[seq_len, seq_len]``

        Returns:
            output Tensor of shape ``[seq_len, batch_size, ntoken]``
        """
        # src = self.embedding(src) * math.sqrt(self.d_model)
        # src = self.pos_encoder(src)

        if src_mask is None:
            """Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
            """
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src)).to(self.device)
        seq,bs,dim=src.shape
        src_mask=torch.ones(seq,seq).double().to(self.device)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output
    

# class PositionalEncoding(nn.Module):

#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)

#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Arguments:
#             x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)
    


# tr_model=TransformerModel(d_model=512,nhead=8,dim_feedforward=2048,nlayers=3)
# src = torch.rand(32, 10, 512)
# out=tr_model(src)