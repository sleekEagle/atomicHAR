import torch.nn as nn
import torch.nn.functional as F
from models.CNN import CNN
from models.Transformer import Transformer
import torch

class AtomicHAR(nn.Module):
    def __init__(self,conf):
        super().__init__()
        self.cnn=CNN(conf).double()
        self.transformer=Transformer(conf)

    def forward(self, x):
        cnn_out=self.cnn(x)
        cnn_out=torch.swapaxes(cnn_out,1,2)
        trans_out=self.transformer(cnn_out)
        return x