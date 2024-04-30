import torch
import torch.nn as nn
import math
from utils import BRelu

class MdehazeNet(nn.Module):
    def __init__(self):
        super(MdehazeNet, self).__init__()
        self.BRelu = BRelu()
        self.Relu = nn.RReLU()

        self.conv1 = nn.Conv2d(3, 6, 3, 1, 2, dilation=2, bias=True)
        self.conv2 = nn.Conv2d(6, 6, 3, 1, 2, dilation=2, bias=True)
        self.conv3 = nn.Conv2d(12, 6, 3, 1, 2, dilation=2, bias=True)
        self.conv4 = nn.Conv2d(18, 6, 3, 1, 2, dilation=2, bias=True)

        self.conv5 = nn.Conv2d(24, 3, 1, 1, 0, bias=True)
        self.conv6 = nn.Conv2d(24, 3, 1, 1, 0, bias=True)

        self.IN1 = nn.InstanceNorm2d(6, affine=True)
        self.IN2 = nn.InstanceNorm2d(6, affine=True)
        self.IN3 = nn.InstanceNorm2d(6, affine=True)
        self.IN4 = nn.InstanceNorm2d(6, affine=True)
        self.IN5 = nn.InstanceNorm2d(3, affine=True)
        self.IN6 = nn.InstanceNorm2d(3, affine=True)

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        self.sa4 = SpatialAttention()
        self.sa5 = SpatialAttention()

        self.aw1 = Aw_block(24)
        self.aw2 = Aw_block(24)

    def forward(self, x):

        x1 = self.conv1(x)
        x1 = self.IN1(x1)
        x1 = self.Relu(x1)

        x2 = self.conv2(x1)
        x2 = self.IN2(x2)
        x2 = self.Relu(x2)
        x2 = self.sa1(x2)

        concat1 = torch.cat((x1, x2), 1)
        x3 = self.conv3(concat1)
        x3 = self.IN3(x3)
        x3 = self.Relu(x3)
        x3 = self.sa2(x3)

        concat2 = torch.cat((x1, x2, x3), 1)
        x4 = self.conv4(concat2)
        x4 = self.IN4(x4)
        x4 = self.Relu(x4)
        x4 = self.sa3(x4)

        concat3 = torch.cat((x1, x2, x3, x4), 1)

        K = self.aw1(concat3)
        K = self.conv5(K)
        K = self.IN5(K)
        K = self.sa4(K)
        K = self.BRelu(K)

        b = self.aw2(concat3)
        b = self.conv6(b)
        b = self.IN6(b)
        b = self.sa5(b)
        b = self.BRelu(b)

        clean_image = 0.5 * ((K * x) - K + b + 1)

        return clean_image

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, 3, 1, 1, dilation=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv1(y)
        y = self.sigmoid(y)
        return x * y

class Aw_block(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(Aw_block, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.conv3 = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y1 = self.max_pool(x)
        y2 = self.avg_pool(x)
        y1 = self.conv1(y1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y2 = self.conv2(y2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = torch.cat((y1, y2), 2)
        y = self.conv3(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y

from thop import profile
model = MdehazeNet()
input = torch.randn(1, 3, 640, 480)
flops, params = profile(model, inputs=(input, ))
print('flops:{}'.format(flops))
print('params:{}'.format(params))
