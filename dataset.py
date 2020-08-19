import os
import cv2
from torch.utils.data import Dataset, DataLoader

class FlixStockDataset(Dataset):

    def __init__(self, args):
        super().__init__()
        self.image_dir = args.image_dir
        self.matte_dir = args.matte_dir
        self.trimap_dir = args.trimap_dir
        self.patch_size = args.patch_size
        self.files = []

        for name in os.listdir(args.image_dir):
            self.files.append(name)
        
        # TODO: Address this.
        # Assume that matte will have the same filename.

    def __getitem__(self, index):
        file_name = self.files[index]

        image_path = os.path.join(self.image_dir, file_name)
        trimap_path = os.path.join(self.trimap_dir, file_name).replace('.jpg', '.png')
        matte_path = os.path.join(self.matte_dir, file_name).replace('.jpg', '.png')

        image = cv2.imread(image_path)
        trimap = cv2.imread(trimap_path)
        matte = cv2.imread(matte_path)

        # cv2.imshow('matte', matte)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        
        image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        trimap = cv2.resize(trimap, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        matte = cv2.resize(matte, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)

        instance = {
            'image': image,
            'trimap': trimap,
            'matte': matte
        }
        return instance
    
    def __len__(self):
        return len(self.files)

class FlixStockDataLoader(DataLoader):

    def __init__(self, args):
        super().__init__(args)
