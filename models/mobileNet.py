import torch
from torch import nn
import torch.nn.functional as F


class MobileNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(MobileNet, self).__init__()
        self.model = nn.Sequential(
            self._conv_bn(input_channels, 32, 2),
            self._conv_dw(32, 64, 1),
            self._conv_dw(64, 128, 2),
            self._conv_dw(128, 128, 1),
            self._conv_dw(128, 256, 2),
            self._conv_dw(256, 256, 1),
            self._conv_dw(256, 512, 2),
            self._top_conv(512, 512, 5),
            self._conv_dw(512, 1024, 2),
            self._conv_dw(1024, 1024, 1),
        )
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, output_channels)

    def forward(self, x):
        x = self.model(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

    def _top_conv(self, in_channel, out_channel, blocks):
        layers = []
        for i in range(blocks):
            layers.append(self._conv_dw(in_channel, out_channel, 1))
        return nn.Sequential(*layers)

    def _conv_bn(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channel, track_running_stats=False),
            nn.ReLU(inplace=True),
        )

    def _conv_dw(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, stride, 1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel, track_running_stats=False),
            nn.ReLU(inplace=False),
        )
