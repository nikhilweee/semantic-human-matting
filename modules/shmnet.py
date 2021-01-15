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
        names = batch['name']
        # (batch, RGB, patch_size, patch_size)
        images = batch['image'].to(self.args.device)

        # T_Net outputs are softmaxed
        # (batch, BUF, patch_size, patch_size)
        pred_trimaps = self.tnet(images)
        pred_softmax = pred_trimaps.softmax(dim=1)
        _, unsure, foreground = torch.split(pred_softmax, 1, dim=1)

        # (batch, RGBBUF, patch_size, patch_size)
        concat = torch.cat([images, pred_trimaps], dim=1)

        if self.training:
            # (batch, 1, patch_size, patch_size)
            gold_trimaps = batch['trimap'].to(self.args.device)

            # Convert to BUF (BG, UNC, FG) classes.
            gold_trimaps = gold_trimaps.repeat(1, 3, 1, 1) * 255
            gold_trimaps[:, 0, :, :] = gold_trimaps[:, 0, :, :].eq(0).float()
            gold_trimaps[:, 1, :, :] = gold_trimaps[:, 1, :, :].eq(128).float()
            gold_trimaps[:, 2, :, :] = gold_trimaps[:, 2, :, :].eq(255).float()

            if self.args.mode != 'end_to_end':
                # Teacher forcing
                _, unsure, foreground = torch.split(gold_trimaps, 1, dim=1)
                # (batch, RGBBUF, patch_size, patch_size)
                concat = torch.cat([images, gold_trimaps], dim=1)
        
        # (batch, 1, patch_size, patch_size)
        alpha_r = self.mnet(concat)

        # (batch, 1, patch_size, patch_size)
        pred_mattes = foreground + unsure * alpha_r

        instance = {
            'name': names,
            'image': images,
            'pred_trimap': pred_trimaps,
            'pred_matte': pred_mattes,
        }

        if self.training:

            # (batch, RGB, patch_size, patch_size)
            gold_mattes = batch['matte'].to(self.args.device)
            # Convert RGB channels in a matte to single channel.
            # (batch, 1, patch_size, patch_size)
            gold_mattes = gold_mattes.mean(dim=1, keepdim=True)

            # Convert gold_trimaps back to single channel
            # (batch, patch_size, patch_size)
            gold_trimaps = gold_trimaps.argmax(dim=1)

            instance['gold_trimap'] = gold_trimaps
            instance['gold_matte'] = gold_mattes

        return instance
    
