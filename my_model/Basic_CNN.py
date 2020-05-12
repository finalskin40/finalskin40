import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Basic_CNN(nn.Module):
    def __init__(self):
        super(Basic_CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 3)
        self.conv3 = nn.Conv2d(128, 256, 3, 3)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(2304, 240)
        self.fc2 = nn.Linear(240, 40)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = self.dropout2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = F.relu(x)

        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
