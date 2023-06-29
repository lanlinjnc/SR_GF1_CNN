#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/29 17:32
# @Author  : lanlin
# 融合空间注意力机制和通道注意力机制
# 参数量：0.94m


import math
import torch
import argparse
import torch.nn as nn
from torch.nn import functional as F


class MY_CNN(nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, num_feat=64, upscale=2, num_resblock=10):
        super(MY_CNN, self).__init__()

        upsample_block_num = int(math.log(upscale, 2))
        self.upscale = upscale

        # 多尺度，不同大小卷积核concatenation
        self.block1_1 = nn.Sequential(
            nn.Conv2d(num_in_ch, 1, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.2)
        )
        self.block1_2 = nn.Sequential(
            nn.Conv2d(num_in_ch, 1, kernel_size=5, padding=2, padding_mode='replicate'),
            nn.LeakyReLU(0.2)
        )
        self.block1_3 = nn.Sequential(
            nn.Conv2d(num_in_ch, 1, kernel_size=7, padding=3, padding_mode='replicate'),
            nn.LeakyReLU(0.2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(3, num_feat, kernel_size=3, padding=1, padding_mode='replicate'),
            # nn.BatchNorm2d(num_feat)
        )

        # 残差网络设计，个数？
        resblocks = [ResidualBlock(num_feat) for _ in range(num_resblock)]
        self.resblocks = nn.Sequential(*resblocks)

        self.block4 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, padding_mode='replicate'),
            # nn.BatchNorm2d(num_feat)
        )

        self.SA = SpatialAttention(kernel_size=3)

        # PixelShuffler放大 选择放大2或者4倍
        block6 = [UpsampleBLock(num_feat, 2) for _ in range(upsample_block_num)]
        self.block6 = nn.Sequential(*block6)

        self.block7 = nn.Conv2d(num_feat, 3, kernel_size=3, padding=1, padding_mode='replicate')
        self.block8 = nn.Conv2d(4, num_out_ch, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x):
        x_up = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        block1_1 = self.block1_1(x)
        block1_2 = self.block1_2(x)
        block1_3 = self.block1_3(x)
        block1 = torch.cat((block1_1, block1_2, block1_3), 1)

        block2 = self.block2(block1)

        block3 = self.resblocks(block2)
        block4 = self.block4(block3)
        block5 = block4 * self.SA(block4)  # 将这个权值乘上原输入特征层
        block6 = self.block6(block2 + block5)
        block7 = self.block7(block6)
        block7 = torch.cat((block7, x_up), 1)
        block8 = self.block8(block7)

        return (torch.tanh(block8) + 1) / 2


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn2 = nn.BatchNorm2d(channels)
        self.SELayer = SELayer(channels, reduction=16)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.lrelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.SELayer(residual)

        return x + residual


# 通道注意力机制，经典SE模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):  # 传入输入通道数，缩放比例
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化高宽为1
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 升维
            nn.Sigmoid())
        self.fc2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b,c,h.w
        b, c, _, _ = x.size()  # batch \channel\ high\ weight
        # b,c,1,1----> b,c
        y = self.avg_pool(x).view(b, c)  # 调整维度、去掉最后两个维度
        # b,c- ----> b,c/16 ---- >b,c ----> b,c,1,1
        y1 = self.fc1(y).view(b, c, 1, 1)  # 添加上h,w维度

        # b,c,1,1----> b,c
        z = self.avg_pool(x)  # 平均欧化
        # b,c- ----> b,c/16 ---- >b,c
        y2 = self.fc2(z)  # 降维、升维

        return x * y1.expand_as(x)  # 来扩展张量中某维数据的尺寸，将输入tensor的维度扩展为与指定tensor相同的size


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        # 经过一个卷积层，输入维度是2，输出维度是1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # sigmoid激活操作

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的平均值  b,1,h,w
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的最大值  b,1,h,w
        x = torch.cat([avg_out, max_out], dim=1)  # 在第一维度上拼接，变为 b,2,h,w
        x = self.conv1(x)  # 转换为维度，变为 b,1,h,w
        return self.sigmoid(x)  # sigmoid激活操作


# PixelShuffler 放大模块
class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1, padding_mode='replicate')
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


if __name__ == "__main__":

    # 初始参数设定
    parser = argparse.ArgumentParser()  # argparse是python用于解析命令行参数和选项的标准模块
    # parser.add_argument('--train-file', type=str, required=True,)  # 训练 h5文件目录
    # parser.add_argument('--eval-file', type=str, required=True)  # 测试 h5文件目录
    # parser.add_argument('--outputs-dir', type=str, required=True)   # 模型 .pth保存目录
    parser.add_argument('--scale', type=int, default=2)  # 放大倍数
    # parser.add_argument('--lr', type=float, default=1e-4)  # 学习率
    # parser.add_argument('--batch-size', type=int, default=356)  # 一次处理的图片大小
    # parser.add_argument('--num-workers', type=int, default=3)  # 线程数
    # parser.add_argument('--num-epochs', type=int, default=50)  # 训练次数
    parser.add_argument('--seed', type=int, default=123)  # 随机种子
    args = parser.parse_args()

    # gpu或者cpu模式，取决于当前cpu是否可用
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 每次程序运行生成的随机数固定
    torch.manual_seed(args.seed)

    test_image_in = torch.ones((1, 1, 128, 128)).to(device)
    MY_CNN = MY_CNN(num_in_ch=1, num_out_ch=1, num_feat=64).to(device)
    test_image_out = MY_CNN(test_image_in)
    print(test_image_out.shape)
