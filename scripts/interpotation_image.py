#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/21 12:23
# @Author  : lanlin

import PIL.Image as pil_image
import numpy
import os


if __name__ == '__main__':
    image_file = 'E:/mydata/PythonCode/datasets/CE2_07m_N/val_hr_192/N008_crop_98.tif'
    scale = 2
    output_dir = 'E:/mydata/PythonCode/SR_comparison/img_results/'
    filename = os.path.basename(image_file).split('.')[0]

    ori_img = pil_image.open(image_file)
    ori_bic = ori_img.resize((ori_img.width * scale, ori_img.height * scale), pil_image.BICUBIC)
    ori_bic.save(os.path.join(output_dir, '{}_ori_x{}_bicubic.png'.format(filename, scale)))
