import torch
import argparse
import logging
import time

from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import FlixStockDataset
from modules.shmnet import SHMNet
from utils import load_checkpoint, save_checkpoint


logging_format = '[%(asctime)-15s] [%(name)s:%(lineno)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger('train')


def calculate_loss(args, outputs):
    image = outputs['image']                # (batch, RGB, patch_size, patch_size)
    pred_matte = outputs['pred_matte']      # (batch, 1, patch_size, patch_size)
    gold_matte = outputs['gold_matte']      # (batch, 1, patch_size, patch_size)
    pred_trimap = outputs['pred_trimap']    # (batch, FUB, patch_size, patch_size)
    gold_trimap = outputs['gold_trimap']    # (batch, 1, patch_size, patch_size)

    pred_comp = pred_matte * image          # (batch, RGB, patch_size, patch_size)
    gold_comp = gold_matte * image          # (batch, RGB, patch_size, patch_size)

    criterion = nn.CrossEntropyLoss()
    alpha_loss = torch.abs(gold_matte - pred_matte).mean()
    comp_loss = torch.abs(gold_comp - pred_comp).mean()
    class_loss = criterion(pred_trimap, gold_trimap)

    if args.mode == 'pretrain_tnet':
        loss = class_loss
    if args.mode == 'pretrain_mnet':
        loss = 0.5 * comp_loss + 0.5 * alpha_loss
    if args.mode == 'end_to_end':
        loss = 0.5 * comp_loss + 0.5 * alpha_loss + 0.01 * class_loss

    return loss


def main(args):

    logger.info(args)

    model = SHMNet(args)
    model.train()
    train_data = FlixStockDataset(args, split='train')
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=1e-4)
    
    start_epoch = 1
    losses = []

    # Load a previous checkpoint if exists
    checkpoint = load_checkpoint(args)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        losses = checkpoint['losses']

    # Start training 
    for epoch in range(start_epoch, args.num_epochs + 1):
        logger.info(f'Epoch: {epoch}/{args.num_epochs}')
        epoch_loss = 0

        for idx, batch in enumerate(train_data_loader):
            outputs = model(batch)
            loss = calculate_loss(args, outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            logger.info(f'Batch: {idx + 1}/{len(train_data_loader)} \t'
                        f'Loss: {epoch_loss / (idx + 1):8.5f}')
        
        average_loss = epoch_loss/(idx + 1)
        losses.append(average_loss)
        if min(losses) == average_loss:
            logger.info('Minimal loss so far.')
            save_checkpoint(args, epoch, losses, model, optimizer, best=True)
        else:
            save_checkpoint(args, epoch, losses, model, optimizer, best=False)            


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train a Semantic Human Matting model.')
    parser.add_argument('--image-dir', type=str, default='data/images',
                        help='Path to directory containing training images.')
    parser.add_argument('--trimap-dir', type=str, default='data/trimaps',
                        help='Path to directory containing intermediate trimaps.')
    parser.add_argument('--matte-dir', type=str, default='data/mattes',
                        help='Path to directory containing final mattes.')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Path to directory containing checkpoints.')
    parser.add_argument('--prefix', type=str, default='shmnet',
                        help='Prefix to add before saving models')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='input batch size for train')
    parser.add_argument('--num-epochs', type=int, default=10,
                        help='number of epochs to train for.')
    parser.add_argument('--learning-rate', type=float, default=1e-5,
                        help='learning rate while training.')
    parser.add_argument('--patch-size', type=int, default=80,
                        help='patch size of input images.')
    parser.add_argument('--mode', type=str, choices=['end_to_end', 'pretrain_mnet', 'pretrain_tnet'],
                        default='end_to_end', help='working mode.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0'], default='cpu',
                        help='device to use.')

    args = parser.parse_args()
    main(args)
