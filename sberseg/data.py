from torch.utils.data import Dataset, DataLoader
import os
import cv2
import torchvision.transforms as transforms
import pytorch_lightning as pl


class UAVidDataset(Dataset):
    def __init__(self, data_path: str, stage = 'test', transform = None):
        self.valid_labels = [0, 33, 38, 56, 75, 79, 90, 113]
        self.void_labels = []
        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_labels, range(8)))
        self.stage = stage
        self.imgs_folder = 'Images'
        self.mask_path = 'Labels'
        self.img_path = f'{data_path}/uavid_{stage}/'
        self.transform = transform
        
        self.img_list = self.get_filenames(self.img_path, self.imgs_folder)
        self.mask_list = None
        if self.stage in ['train', 'val']:
            self.mask_list = self.get_filenames(self.img_path, self.mask_path)
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img = cv2.imread(self.img_list[idx])
        img = cv2.resize(img, (960, 540), )

        if self.transform:
            img = self.transform(img)
            assert(img.shape == (3, 540, 960))
        else:
            assert(img.shape == (540, 960, 3))

        if self.stage in ['train', 'val']:
            mask = cv2.imread(self.mask_list[idx], cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (960, 540), interpolation=cv2.INTER_NEAREST)
            mask = self.encode_segmap(mask)
            assert(mask.shape == (540, 960))

            return img, mask
        
        return img
    
    def encode_segmap(self, mask):
        for voidc in self.void_labels :
            mask[mask == voidc] = self.ignore_index
        for validc in self.valid_labels :
            mask[mask == validc] = self.class_map[validc]
        return mask
    
    def get_filenames(self, path, dir):
        files_list = list()
        for seq in os.listdir(path):
            full_path = os.path.join(path, seq, dir)
            for filename in os.listdir(full_path):
                files_list.append(os.path.join(full_path, filename))
        return files_list


class UAVidDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str = 'data', batch_size: int = 4, num_workers: int = 4):
        super(UAVidDataModule, self).__init__()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.35675976, 0.37380189, 0.3764753], std = [0.32064945, 0.32098866, 0.32325324])
        ])

        self.train_dataset = UAVidDataset(data_path, 'train', transform)
        self.val_dataset = UAVidDataset(data_path, 'val', transform)
        self.test_dataset = UAVidDataset(data_path, 'test', transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            num_workers=self.num_workers,
            shuffle = True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size = 1, 
            num_workers=self.num_workers,
            shuffle = False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size = 1, 
            num_workers=self.num_workers,
            shuffle = False
        )