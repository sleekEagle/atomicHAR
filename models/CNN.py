import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self,conf):
        super().__init__()
        in_channels=conf.data.num_sensors
        self.conv1 = nn.Conv1d(in_channels, 4, 3)
        self.pool = nn.MaxPool1d(2, 2)
        self.conv2 = nn.Conv1d(4, 2, 3)
        self.conv3 = nn.Conv1d(2, 1, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        return x