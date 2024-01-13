import torch.nn as nn
import torch.nn.functional as F

class Linear_decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 40)
        self.fc2 = nn.Linear(40, 40)

    def forward(self, x):
        out=F.relu(self.fc1(x))
        out=F.relu(self.fc2(out))
        return out