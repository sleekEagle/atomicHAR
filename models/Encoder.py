import torch.nn as nn
import torch.nn.functional as F
import torch

#get the padding size required to keep the output of a convolution the same
def get_padding(input_size,dilation,kernel_size,stride):
    output_size=input_size
    return 0.5*(stride*(output_size+1)-input_size+kernel_size+(kernel_size-1)*(dilation-1))

class CNN(nn.Module):
    def __init__(self,in_channels,out_channels,input_size,dilation,kernel_size,stride):
        super().__init__()
        # padding=get_padding(input_size,dilation,kernel_size,stride)
        self.conv1 = nn.Conv1d(in_channels, 16, 
                               kernel_size=3,
                               stride=2,
                               padding=0)
        # self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3,
                               stride=2,
                               padding=0)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3,
                                stride=1,
                                padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class Linear_encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 128)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        return x