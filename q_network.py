import torch.nn as nn


# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        print(f'input {input_size}')
        print(f'output {output_size}')
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)
