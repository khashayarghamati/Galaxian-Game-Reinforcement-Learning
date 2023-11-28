import copy

import torch
from torch import nn

class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.fc_h = nn.Linear(3136, 512)
        self.fc_z = nn.Linear(512, output_size)

    def forward(self, input):
        out = nn.ReLU(self.conv1(input))
        out = nn.ReLU(self.conv2(out))
        out = nn.ReLU(self.conv3(out))
        out = nn.Flatten(out)

        out = nn.ReLU(self.fc_h(out))
        out = self.fc_z(out)
        return out