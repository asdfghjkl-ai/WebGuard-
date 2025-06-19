import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):

        super(ConvBNReLU, self).__init__()

        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]

        # Add batch normalization layer if specified
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))

        # Add ReLU activation function if specified
        if relu:
            conv.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        # Forward pass through the defined layers
        return self.conv(x)


import torch.nn as nn


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()

        self.conv = nn.Sequential(
            ConvBNReLU(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            ConvBNReLU(in_channel, out_channel, k=1, s=1, p=0, relu=relu)
        )

    def forward(self, x):
        # Forward pass through the defined layers
        return self.conv(x)


class TAMM(nn.Module):
    def __init__(self, channel, dilation_level=[1,2,4,8], reduce_factor=4):
        super(TAMM, self).__init__()

        # Initialize the Temporal Attention Module (TAMM)
        self.planes = channel  # Number of input channels
        self.dilation_level = dilation_level  # List of dilation rates for the branches
        self.conv = DSConv3x3(channel, channel, stride=1)  # Initial convolution operation
        self.branches = nn.ModuleList([
            DSConv3x3(channel, channel, stride=1, dilation=d) for d in dilation_level
        ])  # List of branches with dilated convolutions

        self.avg_pool1 = nn.AdaptiveAvgPool2d(1)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(2)
        self.avg_pool4 = nn.AdaptiveAvgPool2d(4)

        # Fully connected layers for feature fusion
        self.fc = nn.Sequential(
            nn.Linear(channel * 21, channel // reduce_factor, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduce_factor, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Initial convolution operation
        conv = self.conv(x)

        brs = [branch(conv) for branch in self.branches]
        brs.append(conv)
        gather = sum(brs)

        # Spatial Pyramid Attention Network (SPAN)
        b, c, _, _ = gather.size()
        y1 = self.avg_pool1(gather).view(b, c)  # Reshape using AdaptiveAvgPool2d
        y2 = self.avg_pool2(gather).view(b, 4 * c)
        y3 = self.avg_pool4(gather).view(b, 16 * c)
        y = torch.cat((y1, y2, y3), 1)

        # Fully connected layers for feature fusion
        y = self.fc(y).view(b, c, 1, 1)

        # Fusion of gated features with the original input through residual connection
        output = gather * y  # Output of Spatial Pyramid Attention Network
        output = output + x  # Residual connection

        return output
