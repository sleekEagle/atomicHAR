import torch.nn as nn
import torch.nn.functional as F

class Linear_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 20)

    def forward(self, x):
        out=F.relu(self.fc1(x))
        out=self.fc2(out)
        return out
    

class CNN_imu_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tconv1 = nn.ConvTranspose1d(8, 8, 3)
        self.tconv2 = nn.ConvTranspose1d(8, 6, 4, dilation=2)
        self.tconv3 = nn.ConvTranspose1d(6, 6, 5, dilation=2)

    def forward(self, x):
        out=F.relu(self.tconv1(x))
        out=F.relu(self.tconv2(out))
        out=self.tconv3(out)
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