import torch
from torch import nn
import torch.nn.functional as F


class mnistCNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(mnistCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, output_channels)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        out = F.log_softmax(x, dim=1)

        return out
