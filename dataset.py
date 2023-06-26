#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/16 19:54
# @Author  : lanlin


import torch
from torch.utils import data
from glob import glob
# import random
from random import randint
from PIL import Image
import os
from torchvision.transforms.functional import to_tensor
import numpy as np
from degradation_file.gray_image_degradation import degradation_bsrgan
import cv2
from torch.nn.functional import interpolate


class dataset(data.Dataset):
    def __init__(self, dirs, patch_size=64, scale=2, is_train=True, bits=10):
        self.patch_size = patch_size
        self.scale = scale
        self.bits = bits

        self.img_list = []
        self.is_train = is_train
        for d in dirs:
            self.img_list = self.img_list + glob(os.path.join(d, '*.tif'))

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img_name = os.path.basename(img_path)

        # img = Image.open(img_path)
        # img = Image.open(img_path).convert('RGB')
        # gt_img = np.asarray(img)

        gt_img = cv2.imread(img_path, -1)
        gt_img, _ = augmentation(gt_img)  # 旋转或者翻转
        lr_img, gt_img = degradation_bsrgan(gt_img/(2**self.bits-1), sf=self.scale, lq_patchsize=self.patch_size)

        lr_img = to_tensor(lr_img).float()
        gt_img = to_tensor(gt_img).float()

        return lr_img, gt_img, img_name


# crop a part of image
def crop_img(img, size, custom=None):
    width, height = size
    if custom is None:
        left = randint(0, img.size[0] - width)
        top = randint(0, img.size[1] - height)
    else:
        left, top = custom

    cropped_img = img.crop((left, top, left + width, top + height))

    return cropped_img, (left, top)


# data augmentation by flipping and rotating
def augmentation(img, custom=None, do_rot=True):
    if custom is None:
        flip_flag = randint(0, 2)
        rot = randint(0, 359)
    else:
        flip_flag, rot = custom
        if rot is None:
            do_rot = False

    # flipping
    if flip_flag == 0:
        img = np.flip(img, 0)
    elif flip_flag == 1:
        img = np.flip(img, 1)
    else:
        pass

    # rotation
    if do_rot:
        if rot < 90:
            rot = 45
            img = np.rot90(img, k=1)
        elif rot < 180:
            rot = 135
            img = np.rot90(img, k=2)
        elif rot < 270:
            rot = 225
            img = np.rot90(img, k=-1)
        else:
            rot = 315
    else:
        rot = None

    return img, (flip_flag, rot)


def discard_boundary(img, ref_size, k_size=None):
    w, h = img.size
    syn_w, syn_h = ref_size  # reference size is given in (W, H)

    if k_size is None:
        w_discard = (w - syn_w) // 2
        h_discard = (h - syn_h) // 2
    else:
        w_discard = k_size // 2
        h_discard = k_size // 2

    w_discard -= 1
    h_discard -= 1
    img = img.crop((w_discard, h_discard, w_discard + syn_w, h_discard + syn_h))

    return img


