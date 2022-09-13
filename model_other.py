import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=1)
    def forward(self, feature):
        x = feature.unsqueeze(dim=2)
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv1(x))
        x = F.dropout(x,training=self.training)
        x = self.conv2(x)
        x = x.squeeze().unsqueeze(dim=1)

        return x

class CNN1(torch.nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=3,padding=1)
    def forward(self, feature):
        x = feature.unsqueeze(dim=0)
        x = F.dropout(x,training=self.training)
        x = self.conv(x)
        x = x.squeeze().unsqueeze(dim=1)

        return x


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.lin1 = Linear(64, 128)     # pan-cancer
        # self.lin1 = Linear(19, 128)     # specific cancer,e.g.luad
        self.lin2 = Linear(128, 32)
        self.lin3 = Linear(32, 1)
    def forward(self, feature):
        x = F.dropout(feature, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, training=self.training)
        x = self.lin3(x)

        return x

