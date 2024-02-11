import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class HARmodel(nn.Module):
    """Model for human-activity-recognition."""
    def __init__(self, num_classes,in_channels,num_features,kernel1_size,kernel2_size):
        super().__init__()
        # Extract features, 1D conv layers
        self.cnn1=nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=kernel1_size,stride=1).double()
        self.cnn2=nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel1_size,stride=1).double()

        self.relu=nn.ReLU()
        self.bn=nn.BatchNorm1d(64).double()
        self.mp=nn.MaxPool1d(kernel_size=2,stride=1)
            
        self.cnn3=nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel1_size,stride=1).double()
        self.cnn4=nn.Conv1d(in_channels=64, out_channels=64, kernel_size=2,stride=1).double()

        self.embedding=nn.Conv1d(in_channels=64, out_channels=num_features, kernel_size=2,stride=1).double()
        self.classifier=nn.Conv1d(in_channels=num_features, out_channels=num_classes, kernel_size=3,stride=1).double()  
        self.num_classes=num_classes  

    def forward(self, x):
        x=self.cnn1(x)
        x=self.cnn2(x)
        x=self.relu(self.bn(x))
        x=self.mp(x)

        x=self.cnn3(x)
        x=self.cnn4(x)
        x=self.relu(self.bn(x))
        x=self.mp(x)

        x=self.relu(self.embedding(x))
        x=self.classifier(x)
        x=torch.mean(x,dim=2)
        output=F.softmax(x,dim=1)
        return output