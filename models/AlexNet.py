import torch.nn as nn
import torch


class AlexNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(AlexNet, self).__init__()
        self.Conv = nn.Sequential(
            # IN : 3*32*32
            nn.Conv2d(in_channels=input_channels, out_channels=96,kernel_size=5,stride=2,padding=2),
            nn.ReLU(),
            # IN : 96*16*16
            nn.MaxPool2d(kernel_size=2,stride=2),
            # IN : 96*8*8
            nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            # IN :256*8*8
            nn.MaxPool2d(kernel_size=2,stride=2),
            # IN : 256*4*4
            nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # IN : 384*4*4
            nn.MaxPool2d(kernel_size=2, stride=2),
            # OUT : 384*2*2
        )
        self.linear = nn.Sequential(
            nn.Linear(in_features=384 * 2 * 2, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=output_channels),
        )

    def forward(self,x):
            x = self.Conv(x)
            x = x.view(-1, 384 * 2 * 2)
            x = self.linear(x)
            return x
