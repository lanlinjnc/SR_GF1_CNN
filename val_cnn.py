import torch
import torchvision.transforms.functional as F
import torch.nn as nn
import os


class G_validation:
    mse = nn.MSELoss()

    def __init__(self, network, loader, writer, save_path):
        self.generator = network
        self.loader = loader
        self.writer = writer
        self.n = loader.dataset.__len__()
        self.save_path = save_path
        self.best = 100

    def run(self, epoch):
        generator = self.generator.eval()
        val_mse_loss = 0

        self.lr_list = []
        self.valid_outputs = []
        self.img_names = []

        for _, val_data in enumerate(self.loader):

            lr, gt, img_name = val_data
            lr = lr.cuda()  # （1,3,192,192）
            gt = gt.cuda()  # （1,3,144,144）

            with torch.no_grad():
                sr = generator(lr)

                self.lr_list.append(lr[0].cpu())
                self.valid_outputs.append(sr[0].cpu())
                self.img_names.append(img_name[0])

                val_mse_loss += self.mse(sr, gt).item()

        val_mse_loss /= self.n
        print("Validation loss(MSE) at %2d:\t==>\t%.6f" % (epoch, val_mse_loss))
        self.writer.add_scalar('G Loss/Total_G_Loss', val_mse_loss, (epoch + 1))
        self.writer.add_scalar('G Loss/HR_loss', val_mse_loss, (epoch + 1))
        self.generator.train()
        if self.best >= val_mse_loss:
            self.best = val_mse_loss
            return True
        else:
            return False


    def save(self, tag):
        save_dir = os.path.join(self.save_path, str(tag))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i in range(self.n):
            lr = self.lr_list[i]
            img = self.valid_outputs[i]
            name = self.img_names[i]

            F.to_pil_image(lr).save(os.path.join(save_dir, 'LR_' + name))
            F.to_pil_image(img).save(os.path.join(save_dir, name))


if __name__=='__main__':
    # define main SR network as generator
    pass
