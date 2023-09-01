import os
import numpy as np
import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import DataLoader, ConcatDataset
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from dataset.cxr_dataset import CxrDataset, CxrBalancedDataset, CxrStudyIdDataset
from dataset.vin_dataset import VinDataset
from dataset.nih_dataset import NihDataset
from dataset.chexpert_dataset import ChexpertDataset
from dataset.transforms import get_transforms


class CxrDataModule(pl.LightningDataModule):
    def __init__(self, datamodule_cfg, dataloader_init_args):
        super(CxrDataModule, self).__init__()
        self.cfg = datamodule_cfg
        self.df = pd.read_csv(self.cfg["train_df_path"])
        self.dataloader_init_args = dataloader_init_args
        if self.cfg["use_pseudo_label"]:
            print("Using pseudo label")
            self.vin_df = pd.read_csv(self.cfg["vinbig_pseudo_train_df_path"])
            self.nih_df = pd.read_csv(self.cfg["nih_pseudo_train_df_path"])
            self.chexpert_df = pd.read_csv(self.cfg["chexpert_pseudo_train_df_path"])

    def setup(self, stage):
        transforms_train, transforms_val = get_transforms(self.cfg["size"])
        if stage in ('fit', 'validate'):
            # split train/val
            msss = MultilabelStratifiedShuffleSplit(
                n_splits=1, test_size=self.cfg["val_split"], random_state=self.cfg["seed"])
            train_idx, val_idx = next(msss.split(
                self.df, self.df[self.cfg["classes"]].values))
            train_df = self.df.iloc[train_idx]
            val_df = self.df.iloc[val_idx]

            self.train_dataset = CxrStudyIdDataset(self.cfg, train_df, transforms_train)
            self.val_dataset = CxrStudyIdDataset(self.cfg, val_df, transforms_val)

            if self.cfg["use_pseudo_label"]:
                vin_dataset = VinDataset(self.cfg, self.vin_df, transforms_train)
                nih_dataset = NihDataset(self.cfg, self.nih_df, transforms_train)
                chexpert_dataset = ChexpertDataset(self.cfg, self.chexpert_df, transforms_train)
                print(f"vin len: {len(vin_dataset)}")
                print(f"nih len: {len(nih_dataset)}")
                print(f"chexpert len: {len(chexpert_dataset)}")
                self.train_dataset = ConcatDataset([self.train_dataset, vin_dataset, nih_dataset, chexpert_dataset])

            print(f"train len: {len(self.train_dataset)}")
            print(f"val len: {len(self.val_dataset)}")

        elif stage == 'predict':
            if self.cfg["predict_pseudo_label"] == "vinbig":
                print("predicting with vinbig dataset")
                pred_df = pd.read_csv(self.cfg["vinbig_train_df_path"])
                self.pred_dataset = VinDataset(self.cfg, pred_df, transforms_val)
            elif self.cfg["predict_pseudo_label"] == "nih":
                print("predicting with nih dataset")
                pred_df = pd.read_csv(self.cfg["nih_train_df_path"])
                self.pred_dataset = NihDataset(self.cfg, pred_df, transforms_val)
            elif self.cfg["predict_pseudo_label"] == "chexpert":
                print("predicting with chexpert dataset")
                pred_df = pd.read_csv(self.cfg["chexpert_train_df_path"])
                self.pred_dataset = ChexpertDataset(self.cfg, pred_df, transforms_val)
            else:
                pred_df = pd.read_csv(self.cfg["pred_df_path"])
                self.pred_dataset = CxrStudyIdDataset(self.cfg, pred_df, transforms_val)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **self.dataloader_init_args, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, **self.dataloader_init_args, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, **self.dataloader_init_args, shuffle=False)