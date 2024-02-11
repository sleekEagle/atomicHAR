import torch.nn as nn
import torch.nn.functional as F
import torch

class Linear_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(992, 1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 1000)

    def forward(self, x):
        out=F.relu(self.fc1(x))
        out=F.relu(self.fc2(out))
        out=self.fc3(out)
        return out
    

class CNN_imu_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv1 = nn.ConvTranspose1d(32, 32, 12,dilation=25)
        self.tconv2 = nn.ConvTranspose1d(32, 16, 12,dilation=23)
        self.tconv3 = nn.ConvTranspose1d(16, 16, 12,dilation=20)
        self.tconv4 = nn.ConvTranspose1d(16, 6, 3,dilation=1)
        self.tconv5 = nn.ConvTranspose1d(6, 6, 11,dilation=16)

        self.conv1 = nn.Conv1d(32, 16, 
                        kernel_size=3,
                        stride=1,
                        padding=0)
        self.conv2 = nn.Conv1d(16, 16, 
                kernel_size=3,
                stride=1,
                padding=0)
        self.conv3 = nn.Conv1d(16, 1, 
                kernel_size=3,
                stride=1,
                padding=0)
        self.lin1= nn.Linear(244, 1000)
        self.lin2= nn.Linear(1000, 1000)

    def forward(self, x):
        out=F.relu(self.tconv1(x))
        out=F.relu(self.tconv2(out))
        out=F.relu(self.tconv3(out))
        out=self.tconv4(out)
        return out
    
class CNN_atom_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv1 = nn.ConvTranspose1d(1, 4, 4,dilation=11)
        self.tconv2 = nn.ConvTranspose1d(4, 4, 4,dilation=11)
        self.tconv3 = nn.ConvTranspose1d(4, 4, 5,dilation=11)
        self.tconv4 = nn.ConvTranspose1d(4, 6, 7,dilation=11)
        self.tconv5 = nn.ConvTranspose1d(6, 6, 3,dilation=4)

    def forward(self, x):
        out=F.relu(self.tconv1(x))
        out=F.relu(self.tconv2(out))
        out=F.relu(self.tconv3(out))
        out=F.relu(self.tconv4(out))
        out=self.tconv5(out)
        return out
    
class Linear_atom_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 200)

    def forward(self, x):
        out=F.relu(self.fc1(x))
        out=self.fc2(out)
        return out
    
class CNN_xyz_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv1 = nn.ConvTranspose1d(8, 8, 3)
        self.tconv2 = nn.ConvTranspose1d(8, 6, 4, dilation=2)
        self.tconv3 = nn.ConvTranspose1d(6, 3, 5, dilation=2)

    def forward(self, x):
        out=F.relu(self.tconv1(x))
        out=F.relu(self.tconv2(out))
        out=self.tconv3(out)
        return out
    

class Activity_classifier_CNN(nn.Module):
    def __init__(self,atom_emb_dim,n_activities):
        super().__init__()
        # padding=get_padding(input_size,dilation,kernel_size,stride)
        self.conv1 = nn.Conv1d(atom_emb_dim, 16, 
                               kernel_size=3,
                               stride=1,
                               padding=0)
        # self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(16, 16, kernel_size=3,
                               stride=1,
                               padding=0)
        self.conv3 = nn.Conv1d(16, 16, kernel_size=3,
                        stride=1,
                        padding=0)
        # self.conv3 = nn.Conv1d(16, n_activities, kernel_size=3,
        #                         stride=1,
        #                         padding=0)
        self.lin=nn.Linear(64, 21)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = self.lin(x)
        x = self.sm(x)
        return x
    

class Activity_classifier_LIN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1=nn.Linear(160, 64)
        self.lin2=nn.Linear(64, 21)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x=torch.flatten(x,1)
        x=F.relu(self.lin1(x))
        x=F.relu(self.lin2(x))
        x=self.sm(x)
        return x