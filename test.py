import time
import torch
import argparse
import logging

from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import FlixStockDataset
from modules.shmnet import SHMNet
from utils import load_checkpoint, save_images


logging_format = '[%(asctime)-15s] [%(name)s:%(lineno)s] %(message)s'
logging.basicConfig(level=logging.INFO, format=logging_format)
logger = logging.getLogger('test')


def main(args):

    logger.info(args)

    model = SHMNet(args)
    model.eval()
    test_data = FlixStockDataset(args, split='test')
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size)

    # Load a previous checkpoint if exists
    checkpoint = load_checkpoint(args)
    if not checkpoint:
        raise ValueError('A checkpoint is required to evaluate.')
    model.load_state_dict(checkpoint['model_state_dict'])

    # Start testing 
    for idx, batch in enumerate(test_data_loader):
        outputs = model(batch)
        logger.info(f'Batch: {idx + 1}/{len(test_data_loader)}')
        save_images(args, outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Test a Semantic Human Matting model.')
    parser.add_argument('--image-dir', type=str, default='data/images',
                        help='Path to directory containing testing images.')
    parser.add_argument('--trimap-dir', type=str, default='data/trimaps',
                        help='Path to directory containing intermediate trimaps.')
    parser.add_argument('--matte-dir', type=str, default='data/mattes',
                        help='Path to directory containing final mattes.')
    parser.add_argument('--save-dir', type=str, default='data/predictions',
                        help='Path to directory containing final predictions.')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Path to directory containing checkpoints.')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='input batch size for train')
    parser.add_argument('--prefix', type=str, default='shmnet',
                        help='Prefix to look for before loading models')
    parser.add_argument('--patch-size', type=int, default=200,
                        help='patch size of input images.')
    parser.add_argument('--mode', type=str, choices=['end_to_end', 'pretrain_mnet', 'pretrain_tnet'],
                        default='end_to_end', help='working mode.')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda:0'], default='cpu',
                        help='device to use.')

    args = parser.parse_args()
    main(args)
