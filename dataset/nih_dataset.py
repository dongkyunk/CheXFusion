import os
import cv2
import numpy as np
from torch import from_numpy
from torch.utils.data import Dataset


class NihDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.iloc[index][self.cfg['classes']].to_numpy().astype(np.float32)    
        path = os.path.join(self.cfg['data_dir'], "nih/images_001/images", self.df.iloc[index]["id"])

        img = cv2.imread(path)
        assert img.shape == (self.cfg['size'], self.cfg['size'], 3), f"{img.shape}"

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']   
            img = np.moveaxis(img, -1, 0)

        return img, label