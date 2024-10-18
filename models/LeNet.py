import torch
from torch import nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 6, 5),  # in_channels, out_channels, kernel_size
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, output_channels)
        )

    def forward(self, X):
        feature = self.conv(X)
        output = self.fc(feature.view(X.shape[0], -1))
        return output
