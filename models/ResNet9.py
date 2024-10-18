import torch
from torch import nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, pool=False, pool_no=2):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels, track_running_stats=False),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(pool_no))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ResNet9, self).__init__()

        self.conv1 = conv_block(input_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 256, pool=True)
        self.res2 = nn.Sequential(conv_block(256, 256), conv_block(256, 256))

        self.MP = nn.MaxPool2d(2)
        self.FlatFeats = nn.Flatten()
        self.classifier = nn.Linear(1024, output_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x) + x
        x = self.MP(x)
        x = self.FlatFeats(x)
        out = self.classifier(x)
        return out
