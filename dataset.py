import os
import cv2
from torch.utils.data import Dataset, DataLoader

class FlixStockDataset(Dataset):

    def __init__(self, args, split='train'):
        super().__init__()
        self.image_dir = args.image_dir
        self.matte_dir = args.matte_dir
        self.trimap_dir = args.trimap_dir
        self.patch_size = args.patch_size
        self.split = split
        self.files = []

        for name in os.listdir(args.image_dir):
            self.files.append(name)
        
        if split == 'train':
            self.files = self.files[:90]
        if split == 'test':
            self.files = self.files[:10]

    def __getitem__(self, index):
        file_name = self.files[index]

        image_path = os.path.join(self.image_dir, file_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)

        instance = {
            'name': file_name,
            'image': image
        }

        if self.split == 'train':
            trimap_path = os.path.join(self.trimap_dir, file_name).replace('.jpg', '.png')
            trimap = cv2.imread(trimap_path)
            trimap = cv2.resize(trimap, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)

            matte_path = os.path.join(self.matte_dir, file_name).replace('.jpg', '.png')
            matte = cv2.imread(matte_path)
            matte = cv2.resize(matte, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)

            instance['trimap'] = trimap
            instance['matte'] = matte

        return instance
    
    def __len__(self):
        return len(self.files)

