import copy

import torch
from torch import nn


class Model(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        c, h, w = input_size

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.fc_h = nn.Linear(3136, 512)
        self.fc_z = nn.Linear(512, output_size)

    def forward(self, input):
        input = nn.functional.relu(self.conv1(input))
        input = nn.functional.relu(self.conv2(input))
        input = nn.functional.relu(self.conv3(input))

        q = input.view(-1, 3136)

        q = nn.functional.relu(self.fc_h(q))
        q = self.fc_z(q)
        return q