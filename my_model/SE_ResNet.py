
from torch import nn
import torch
import torch.nn as nn
import torch.nn.functional as F


# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        Initializes internal Module state, shared by both nn.Module.
        """
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # To-Do:add code here

        self.downsample = downsample  # 我们在两次卷积中可能会使输入的tensor的size与输出的tensor的size不相等,
        # 为了使它们能够相加,所以输出的tensor与输入的tensor size不同时,
        # 我们使用downsample(由外部传入)来使保持size相同

    def forward(self, x):
        """
        Defines the computation performed at every call.
        x: N * C * H * W
        """
        # if the size of input x changes, using downsample to change the size of residual
        out = self.left(x)
        residual = x
        if self.downsample:
            residual = self.downsample(x)
        out = F.relu(out)
        # To-Do:add  code here

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        """
        Initializes internal Module state, shared by both nn.Module and ScriptModule.
        """
        super(ResNet, self).__init__()
        self.in_channels = 64

        # part1
        # To-Do:intialize part 1 here
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # part2
        self.layer1 = self.make_layer(block, 64, num_blocks=layers[0])
        self.layer2 = self.make_layer(block, 128, num_blocks=layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks=layers[2],
                                      stride=2)  # To-Do: complete defination of self.layer3
        self.layer4 = self.make_layer(block, 512, num_blocks=layers[3], stride=2)

        # part3
        self.fc = nn.Linear(8192, 40)
        # To-Do: initialize part 3 here

    def make_layer(self, block, out_channels, num_blocks, stride=1):
        """
        make a layer with num_blocks blocks.
        """
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            # use Conv2d with stride to downsample
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))

        # first block with downsample
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))

        self.in_channels = out_channels
        # add num_blocks - 1 blocks
        for i in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))

        # return a layer containing layers
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Defines the computation performed at every call.
        """
        # To-Do: define the computation of  ResNet
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # view: here change output size from 4 dimensions to 2 dimensions
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SELayer(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class SEResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(SEResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.se = SELayer(out_channels, reduction)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se(out) + residual
        out = self.relu(out)

        return out
