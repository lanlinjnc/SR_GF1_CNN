#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
1. read the whole files under a certain folder
2. chose 10000 files randomly
3. copy them to another folder and save
"""


import os
import random
random.seed(1)
import shutil


def copyFile(fileDir, tarDir):
    # 1
    pathDir = os.listdir(fileDir)

    # 2
    sample = random.sample(pathDir, 924)
    # print(sample)

    # 3
    for name in sample:
        ori_file_name = fileDir + name
        shutil.copyfile(ori_file_name, tarDir + name)
        os.remove(ori_file_name)
        print(ori_file_name)


if __name__ == '__main__':
    train_dir = "E:/mydata/PythonCode/SR_GF1_CNN/datasets/GFDM01_PAN_16bit_1m/train/"
    val_dir = 'E:/mydata/PythonCode/SR_GF1_CNN/datasets/GFDM01_PAN_16bit_1m/val/'
    copyFile(train_dir, val_dir)
