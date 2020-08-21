import os
import cv2
import logging
import transforms
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger('dataset')


class SHMDataset(Dataset):

    def __init__(self, args, split='train'):
        super().__init__()
        self.image_dir = args.image_dir
        self.matte_dir = args.matte_dir
        self.trimap_dir = args.trimap_dir
        self.patch_size = args.patch_size
        self.mode = args.mode
        self.split = split
        self.files = []
        self.create_transforms()

        for name in os.listdir(args.image_dir):
            self.files.append(name)
        
        if split == 'train':
            self.files = self.files[:90]
        if split == 'test':
            self.files = self.files[:10]
    
    def create_transforms(self):

        transforms_list = []

        if self.mode == 'pretrain_tnet':
            transforms_list.extend([
                transforms.RandomCrop(400),
                transforms.RandomRotation(180),
                transforms.RandomHorizontalFlip()
            ])
        if self.mode == 'pretrain_mnet':
            transforms_list.extend([
                transforms.RandomCrop(320),
            ])
        if self.mode == 'end_to_end':
            transforms_list.extend([
                transforms.RandomCrop(800),
            ])

        transforms_list.extend([
            transforms.Resize((self.patch_size, self.patch_size)),
            transforms.ToTensor()
        ])
        
        self.transforms = transforms.Compose(transforms_list)


    def __getitem__(self, index):
        file_name = self.files[index]

        image_path = os.path.join(self.image_dir, file_name)
        image = Image.open(image_path)

        instance = {'name': file_name}

        if self.split == 'train':
            trimap_path = os.path.join(self.trimap_dir, file_name).replace('.jpg', '.png')
            matte_path = os.path.join(self.matte_dir, file_name).replace('.jpg', '.png')

            trimap = Image.open(trimap_path).convert('L')
            matte = Image.open(matte_path).convert('L')

            [image, trimap, matte] = self.transforms([image, trimap, matte])

            instance['image'] = image
            instance['trimap'] = trimap
            instance['matte'] = matte

        else:
            [image] = self.transforms([image])
            instance['image'] = image

        return instance
    
    def __len__(self):
        return len(self.files)

