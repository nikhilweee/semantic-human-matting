import os
import time
import torch
import logging
from torchvision.utils import save_image

logger = logging.getLogger('utils')


def save_checkpoint(args, epoch, loss, model, optimizer, best=False):
    """Save a checkpoint for future use."""

    checkpoint = {
        'epoch': epoch,
        'loss': loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    if best:
        name = f'{args.prefix}-best-epoch-{epoch}-{int(time.time())}.pt'
    else:
        name = f'{args.prefix}-epoch-{epoch}-{int(time.time())}.pt'

    os.makedirs(args.checkpoint_dir, exist_ok=True)    
    path = os.path.join(args.checkpoint_dir, name)
    logger.info(f'Saving checkpoint to "{path}"')
    torch.save(checkpoint, path)


def load_checkpoint(args):
    """Fetch and load the best checkpoint if it exists."""

    best_model = None
    all_models, best_models = [], []

    for name in os.listdir(args.checkpoint_dir):
        if name.startswith(args.prefix):
            if 'best' in name:
                best_models.append(name)
            else:
                all_models.append(name)

    if best_models:
        best_models.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
        best_model = best_models[-1]
    elif all_models:
        all_models.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
        best_model = all_models[-1]

    if best_model:
        path = os.path.join(args.checkpoint_dir, best_model)
        logger.info(f'Loading checkpoint from "{path}"')
        checkpoint = torch.load(path)
        return checkpoint
    
    return None


def save_images(args, outputs):
    """Save a batch of images."""

    if not os.path.isdir(args.save_dir):
        os.makedirs(args.save_dir)

    logger.info(f'Saving {len(outputs["name"])} images to {args.save_dir}')

    for idx, name in enumerate(outputs['name']):
        matte = outputs['pred_matte'][idx]
        save_path = os.path.join(args.save_dir, name)
        save_image(matte, save_path)
