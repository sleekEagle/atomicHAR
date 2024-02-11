import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import nn, Tensor
import math

#get the padding size required to keep the output of a convolution the same
def get_padding(input_size,dilation,kernel_size,stride):
    output_size=input_size
    return 0.5*(stride*(output_size+1)-input_size+kernel_size+(kernel_size-1)*(dilation-1))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: any tensor with positions as decimal numbers
        """
        pos_enc=self.pe[x,0,:]
        return self.dropout(pos_enc)
    

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        # padding=get_padding(input_size,dilation,kernel_size,stride)
        self.conv1 = nn.Conv1d(6, 16, 
                               kernel_size=3,
                               stride=2,
                               padding=0)
        self.mp1 = nn.MaxPool1d(3, stride=1,padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3,
                               stride=2,
                               padding=0)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=3,
                                stride=2,
                                padding=0)
        self.conv4 = nn.Conv1d(16, 16, kernel_size=1,
                        stride=2,
                        padding=0)
        self.ts_pos_encoder = PositionalEncoding(32)
        self.atom_pos_encoder = PositionalEncoding(8)

    def forward(self, x):
        out=F.relu(self.conv1(x))
        out=F.relu(self.conv2(out))
        out=F.relu(self.conv3(out))
        out=F.relu(self.conv4(out))

        # argmax_win_len=10
        # thr=0.5

        # bs,dim,l=conv2_out.shape
        # max_atoms=torch.argmax(conv2_out,dim=1)
        # atom_emb=self.atom_pos_encoder(max_atoms)
        # pos=torch.arange(0,l)
        # pos_emb=self.ts_pos_encoder(pos).unsqueeze(dim=0).repeat(bs,1,1)
        # pos_atom_emb=torch.cat((atom_emb,pos_emb),dim=2)

        return out

class Linear_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 128)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        return x
    

class Atom_encoder_CNN(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        # padding=get_padding(input_size,dilation,kernel_size,stride)
        self.conv1 = nn.Conv1d(32, 16, 
                               kernel_size=3,
                               stride=2,
                               padding=0)
        # self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, out_channels, kernel_size=3,
                               stride=2,
                               padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x[:,:,0]