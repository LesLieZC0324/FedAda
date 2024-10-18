import torch
from torch import nn


class mnistLR(nn.Module):
    def __init__(self, input_channels=1, output_channels=10):
        super(mnistLR, self).__init__()
        self.linear = nn.Linear(input_channels * 28 * 28, output_channels)

    def forward(self, x):
        x = x.view(-1, 784)
        output = self.linear(x)
        return output
