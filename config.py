# PerPix_x2  输入图像192  'patch size': 96  'scale': 2,  'batch size': 4
# PerPix_x3  输入图像192  'patch size': 64  'scale': 3,  'batch size': 8
# PerPix_x4  输入图像192  'patch size': 48  'scale': 4,  'batch size': 16

name = 'net_cnn1_x2'  # configurate name for the model: used for saving ckpt, validation, and logs
config = {
  'train': {
    'patch_size': 128,
    'batch_size': 32,
    'num_workers': 0,
    'iterations_G': 1500,  # for G: 300000, for C: 200000
    'lr_G': 2e-4,
    'decay': {
      'every': 10000,
      'by': 0.1
    },
  },

  'valid': {
    'batch_size': 1,
    'every': 5,  # 同时每2个epoch保存一次模型
  },

  'model': {
    'scale': 2,
    'kernel_size': 64,
    'bits': 10,
  },

  'path': {
    'project': '/project',
    'ckpt': './ckpt/{}.pth'.format(name),
    'dataset': {
      'train': ['E:/mydata/PythonCode/SR_GF1_CNN/datasets/GF2_PAN2_16bit_1m/train/'],
      # 'train': ['/dataset/DIV2K/train_HR',
      #           '/dataset/Flickr2K/Flickr2K_HR'],
      'valid': ['E:/mydata/PythonCode/SR_GF1_CNN/datasets/GF2_PAN2_16bit_1m/val/'],
    },

    'validation': './validation/{}'.format(name),
    'logs': './logs/{}'.format(name)
  }
}