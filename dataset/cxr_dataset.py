import os
import cv2
import pandas as pd
import numpy as np
import jpeg4py as jpeg
from torch import from_numpy
from torch.utils.data import Dataset


class CxrDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if all([c in self.df.columns for c in self.cfg['classes']]):
            label = self.df.iloc[index][self.cfg['classes']].to_numpy().astype(np.float32)    
        else:
            label = np.zeros(len(self.cfg['classes']))

        path = self.df.iloc[index]["path"]
        path = os.path.join(self.cfg['data_dir'], path)
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


class CxrBalancedDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        class_name = self.cfg['classes'][index%len(self.cfg['classes'])]
        df = self.df[self.df[class_name] == 1].sample(1).iloc[0]

        label = df[self.cfg['classes']].to_numpy().astype(np.float32)    

        path = df["path"]
        path = os.path.join(self.cfg['data_dir'], path)
        resized_path = path.replace(".jpg", f"_resized_{self.cfg['size']}.jpg")

        if os.path.exists(resized_path):
            img = jpeg.JPEG(resized_path).decode()
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


class CxrStudyIdDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df.groupby("study_id")
        self.study_ids = list(self.df.groups.keys())
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        df = self.df.get_group(self.study_ids[index])
        if len(df) > 4:
            df = df.sample(4)

        if all([c in df.columns for c in self.cfg['classes']]):
            label = df[self.cfg['classes']].iloc[0].to_numpy().astype(np.float32)    
        else:
            label = np.zeros(len(self.cfg['classes']))

        imgs = []
        for i in range(len(df)):
            path = df.iloc[i]["path"]
            path = os.path.join(self.cfg['data_dir'], path)
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
            imgs.append(img)

        img = np.stack(imgs, axis=0)    
        img = np.concatenate([img, np.zeros((4-len(df), 3, self.cfg['size'], self.cfg['size']))], axis=0).astype(np.float32)
        return img, label

