import torch


class Model:
    def __init__(self, input_size, output_size):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_size, out_channels=32, kernel_size=(8, 8), stride=4),
            torch.nn.ReLU())
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(4, 4), stride=2),
            torch.nn.ReLU())
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=1),
            torch.nn.ReLU())
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(in_features=7 * 7 * 64, out_features=256),
            torch.nn.ReLU())
        self.fc2 = torch.nn.Linear(in_features=256, out_features=output_size)

    def forward(self, observation):
        out1 = self.conv1(observation)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.fc1(out3.view(-1, 7 * 7 * 64))
        out = self.fc2(out4)

        return out