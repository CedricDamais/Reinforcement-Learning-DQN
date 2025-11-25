import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=input_shape[0], out_channels=16, kernel_size=8, stride=4
        )

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

        self.fc1 = nn.Linear(32 * 9 * 9, 256)

        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))

        return self.fc2(x)
