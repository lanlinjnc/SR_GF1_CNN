#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/13 21:22
# @Author  : lanlin
# GeoEye卫星的残差卷积网络


import math
import torch
import argparse
import torch.nn as nn


class MY_CNN_NEO(nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, num_feat=64, upscale=2):

        super(MY_CNN_NEO, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, kernel_size=7, padding=3, padding_mode='replicate'),
            nn.LeakyReLU(0.2)
        )
        self.resblock1 = ResidualBlock(num_feat)
        self.resblock2 = ResidualBlock(num_feat)
        self.resblock3 = ResidualBlock(num_feat)
        self.resblock4 = ResidualBlock(num_feat)
        self.resblock5 = ResidualBlock(num_feat)
        self.block7 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=7, padding=3, padding_mode='replicate'),
            nn.BatchNorm2d(num_feat)
        )
        self.block8 = nn.Conv2d(num_feat, num_out_ch, kernel_size=7, padding=3, padding_mode='replicate')

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.resblock1(block1)
        block3 = self.resblock2(block2)
        block4 = self.resblock3(block3)
        block5 = self.resblock4(block4)
        block6 = self.resblock5(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

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

    test_image_in = torch.ones((1, 1, 256, 256)).to(device)
    MY_CNN = MY_CNN_NEO().to(device)
    test_image_out = MY_CNN(test_image_in)
    print(test_image_out.shape)
