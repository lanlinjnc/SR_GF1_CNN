#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/13 21:22
# @Author  : lanlin
# neo卫星：通道注意力机制
# 参数量：0.41m


import torch
import argparse
import torch.nn as nn


class MY_CNN(nn.Module):
    def __init__(self, num_in_ch=1, num_out_ch=1, num_feat=64, upscale=2):

        super(MY_CNN, self).__init__()
        self.upscale = upscale
        self.block1 = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.2)
        )
        self.resblock1 = ResidualBlock(num_feat)
        self.resblock2 = ResidualBlock(num_feat)
        self.resblock3 = ResidualBlock(num_feat)
        self.resblock4 = ResidualBlock(num_feat)
        self.resblock5 = ResidualBlock(num_feat)
        self.block7 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(num_feat)
        )
        self.block8 = nn.Conv2d(num_feat, num_out_ch, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
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

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.lrelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual


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
    MY_CNN = MY_CNN().to(device)
    test_image_out = MY_CNN(test_image_in)
    print(test_image_out.shape)
