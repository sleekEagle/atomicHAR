import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    def __init__(self,conf):
        super().__init__()
        transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x