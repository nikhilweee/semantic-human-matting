import torch

from torch import nn
from modules.pspnet import PSPNet
from modules.dimnet import DIMNet

class SHMNet(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tnet = PSPNet()
        self.mnet = DIMNet()
        self.to(args.device)
    
    def forward(self, batch):
        # (batch, patch_size, patch_size, RGB)
        images = batch['image'].to(self.args.device)
        gold_trimaps = batch['trimap'].to(self.args.device)
        gold_mattes = batch['matte'].to(self.args.device)

        # (batch, RGB, patch_size, patch_size)
        images = images.transpose(3, 1).float()

        # Convert RGB channels to FUB (FG, UNC, BG) classes.
        gold_trimaps[:, :, :, 0] = gold_trimaps[:, :, :, 0].eq(0)
        gold_trimaps[:, :, :, 1] = gold_trimaps[:, :, :, 1].eq(128)
        gold_trimaps[:, :, :, 2] = gold_trimaps[:, :, :, 2].eq(255)
        # (batch, FUB, patch_size, patch_size)
        gold_trimaps = gold_trimaps.transpose(3, 1).float()

        # Convert RGB channels in a matte to single channel.
        gold_mattes = gold_mattes.transpose(3, 1).float()
        # (batch, 1, patch_size, patch_size)
        gold_mattes = gold_mattes.mean(dim=1, keepdim=True)

        # T_Net outputs are softmaxes
        # (batch, FUB, patch_size, patch_size)
        pred_trimaps = self.tnet(images)
        _, unsure, foreground = torch.split(pred_trimaps, 1, dim=1)

        # TODO: Replace gold_trimaps with pred_trimaps
        # (batch, RGBFUB, patch_size, patch_size)
        concat = torch.cat([images, gold_trimaps], dim=1)
        # (batch, 1, patch_size, patch_size)
        alpha_r = self.mnet(concat)

        # Convert gold_trimaps back to single channel
        # (batch, patch_size, patch_size)
        gold_trimaps = gold_trimaps.argmax(dim=1)

        # (batch, 1, patch_size, patch_size)
        pred_mattes = foreground + unsure * alpha_r

        instance = {
            'image': images,
            'pred_trimap': pred_trimaps,
            'gold_trimap': gold_trimaps,
            'pred_matte': pred_mattes,
            'gold_matte': gold_mattes
        }

        return instance
    