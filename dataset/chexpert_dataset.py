import os
import cv2
import numpy as np
import jpeg4py as jpeg
from torch import from_numpy
from torch.utils.data import Dataset


class ChexpertDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        label = self.df.iloc[index][self.cfg['classes']].to_numpy().astype(np.float32)    
        path = os.path.join(self.cfg['data_dir'], 'chexpert/chexpertchestxrays-u20210408', self.df.iloc[index]["Path"])
        resized_path = path.replace(".jpg", f"_resized_{self.cfg['size']}.jpg")

        if os.path.exists(resized_path):
            img = jpeg.JPEG(resized_path).decode()
            if os.path.exists(path):
                os.remove(path)
            assert img.shape == (self.cfg['size'], self.cfg['size'], 3)
        else:
            img = jpeg.JPEG(path).decode()
            img = cv2.resize(img, (self.cfg['size'], self.cfg['size']), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(resized_path, img)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']   
            img = np.moveaxis(img, -1, 0)

        return img, label