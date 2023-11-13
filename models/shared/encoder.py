import torch
from torch import nn
from torch.nn import functional as F

class SimpleEncoder(nn.Module):
    """
    Baseline CNN Encoder as described in Locatello et al. (2020):
    https://arxiv.org/pdf/2006.15055.pdf
    """

    def __init__(self,
                 c_in: int,
                 c_out: int,
                 kernel_size: tuple = (5, 5),
                 stride: tuple = (1, 1),
                 num_encoder_layers: int = 4):
        super().__init__()

        blocks = []
        for i in range(num_encoder_layers):
            c_in = c_in if i == 0 else c_out

            if stride == (1, 1):
                padding = "same"
            elif stride == (1, 2):
                padding = (2, 2)
            else:
                if kernel_size == 5 or kernel_size == (5, 5):
                    padding = (2, 2)
                elif kernel_size == 3 or kernel_size == (3, 3):
                    padding = (1, 1)

            blocks.append(
                nn.Sequential(
                    nn.Conv2d(c_in, out_channels=c_out,
                              kernel_size=kernel_size,
                              padding=padding,
                              stride=stride),
                    nn.ReLU()
                )
            )
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=(1, 1), group_norm = True):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(32, planes, eps=1e-06) if group_norm else nn.Sequential()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(32, planes, eps=1e-06) if group_norm else nn.Sequential()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                kernel_size=1, stride=stride, bias=False),
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,
                 c_in,
                 num_blocks,
                 stride,
                 num_classes=None,
                 group_norm=True):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(c_in, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(32, 64, eps=1e-06) if group_norm else nn.Sequential()

        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks[1], stride=stride)
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks[2], stride=stride)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks[3], stride=stride)

        self.num_classes = num_classes

        if num_classes is not None:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512, num_classes)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        if self.num_classes is not None:
            out = self.avg_pool(out)
            out = torch.flatten(out, 1)
            out = self.fc(out)
            
        return out


def ResNet18(c_in, stride, group_norm=False):
    return ResNet(c_in, [2, 2, 2, 2], stride=stride, group_norm=group_norm)


def ResNet34(c_in, stride, group_norm=False):
    return ResNet(c_in, [3, 4, 6, 3], stride=stride, group_norm=group_norm)
