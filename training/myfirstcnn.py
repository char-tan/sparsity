import torch 
import torch.nn as nn
import torch.nn.functional as F


class MyFirstCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv0 = nn.Conv2d(3, 16, (3, 3))
        self.conv1 = nn.Conv2d(16, 16, (3, 3))
        self.conv2 = nn.Conv2d(16, 3, (3, 3))
        self.linear = nn.Linear(2352, 10)

    def forward(self, x):

        x = F.relu(self.conv0(x))
        x = F.relu(self.conv1(x).view(x.shape[0], -1))
        x = F.relu(self.conv2(x).view(x.shape[0], -1))
        x = F.relu(self.linear(x))

        return x
