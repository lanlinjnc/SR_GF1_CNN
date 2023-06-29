#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/6/8 15:47
# @Author  : lanlin


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from tqdm import tqdm
from tensorboardX import SummaryWriter

from config import config as _config
from dataset import dataset
from val_cnn import G_validation

from networks.net_cnn0 import MY_CNN as Generator


def train(config, epoch_from=0):
    dataParallel = False

    print('process before training...')
    train_dataset = dataset(dirs=config['path']['dataset']['train'], patch_size=config['train']['patch_size'],
                            scale=config['model']['scale'], bits=config['model']['bits'])
    train_data = DataLoader(
        dataset=train_dataset, batch_size=config['train']['batch_size'],
        shuffle=True, num_workers=config['train']['num_workers']
    )

    valid_dataset = dataset(config['path']['dataset']['valid'], patch_size=config['train']['patch_size'],
                            scale=config['model']['scale'], is_train=False)
    valid_data = DataLoader(dataset=valid_dataset, batch_size=config['valid']['batch_size'], num_workers=0)

    # training details - epochs & iterations
    iterations_per_epoch = len(train_dataset) // config['train']['batch_size'] + 1
    n_epoch = config['train']['iterations_G'] // iterations_per_epoch + 1
    print('epochs scheduled: %d , iterations per epoch: %d...' % (n_epoch, iterations_per_epoch))

    # define main SR network as generator
    generator = Generator(num_in_ch=1, num_out_ch=1, num_feat=64, upscale=config['model']['scale'])
    if dataParallel:
        generator = nn.DataParallel(generator)
    generator = generator.cuda()

    save_path_G = config['path']['ckpt']
    save_path_Opt = save_path_G[:-4] + 'Opt.pth'

    # optimizer
    learning_rate = config['train']['lr_G']
    G_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(G_optimizer, config['train']['decay']['every'],
                                             config['train']['decay']['by'])

    # if training from scratch, remove all validation images and logs
    if epoch_from == 0:
        if os.path.exists(config['path']['validation']):
            _old = os.listdir(config['path']['validation'])
            for f in _old:
                if os.path.isfile(os.path.join(config['path']['validation'], f)):
                    os.remove(os.path.join(config['path']['validation'], f))
        if os.path.exists(config['path']['logs']):
            _old = os.listdir(config['path']['logs'])
            for f in _old:
                if os.path.isfile(os.path.join(config['path']['logs'], f)):
                    os.remove(os.path.join(config['path']['logs'], f))

    # if training not from scratch, load weights 如果不是从头开始
    else:
        if os.path.exists(save_path_G):
            print('reading generator checkpoints...')
            generator.load_state_dict(torch.load(save_path_G))
            print('reading optimizer checkpoints...')
            G_optimizer.load_state_dict(torch.load(save_path_Opt))
            lr_scheduler.last_epoch = epoch_from * iterations_per_epoch
        else:
            raise FileNotFoundError('Pretrained weight not found.')

    if not os.path.exists(config['path']['validation']):
        os.makedirs(config['path']['validation'])
    if not os.path.exists(os.path.dirname(config['path']['ckpt'])):
        os.makedirs(os.path.dirname(config['path']['ckpt']))
    if not os.path.exists(config['path']['logs']):
        os.makedirs(config['path']['logs'])
    writer = SummaryWriter(config['path']['logs'])

    # loss functions
    loss = nn.L1Loss().cuda()

    # validation
    valid = G_validation(generator, valid_data, writer, config['path']['validation'])

    # training
    # print(generator)
    # total = sum([param.nelement() for param in generator.parameters()])
    # print("Number of parameter: %.2fM" % (total / 1e6))
    print('start training...')
    for epoch in range(epoch_from, n_epoch):
        generator = generator.train()
        epoch_loss = 0
        train_bar = tqdm(train_data)
        # for i, data in enumerate(train_data):
        for lr, gt, img_name in train_bar:

            lr = lr.cuda()  # (8,3,301,301)
            gt = gt.cuda()  # (8,3,256,256)
            # img_name = img_name.cuda()

            # forwarding
            sr = generator(lr)  # (8,3,128,128)
            g_loss = loss(sr, gt)

            # back propagation
            G_optimizer.zero_grad()
            g_loss.backward()
            G_optimizer.step()
            lr_scheduler.step()
            epoch_loss += g_loss.item()  # 这是一整个epoch的loss和，不是每张图片的loss

        print('Training loss at {:d} : {:.8f}'.format(epoch, epoch_loss))

        # validation
        if (epoch + 1) % config['valid']['every'] == 0:
            is_best = valid.run(epoch + 1)

            # save validation image
            valid.save(tag='latest')
            if is_best:
                if dataParallel:
                    torch.save(generator.module.state_dict(), save_path_G)
                else:
                    torch.save(generator.state_dict(), save_path_G)
            torch.save(G_optimizer.state_dict(), save_path_Opt)


    # training process finished.
    # final validation and save checkpoints
    is_best = valid.run(n_epoch)
    valid.save(tag='final')
    writer.close()
    if is_best:
        if dataParallel:
            torch.save(generator.module.state_dict(), save_path_G)
        else:
            torch.save(generator.state_dict(), save_path_G)
    torch.save(G_optimizer.state_dict(), save_path_Opt)

    print('training finished.')


if __name__ == '__main__':
    train(_config, epoch_from=0)
