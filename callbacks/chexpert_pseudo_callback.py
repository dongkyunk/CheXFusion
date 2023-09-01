import numpy as np
import torch
import pandas as pd
from lightning.pytorch.callbacks import BasePredictionWriter


class ChexpertWriter(BasePredictionWriter):
    def __init__(self, chexpert_train_df_path, chexpert_pseudo_train_df_path, write_interval):
        super().__init__(write_interval)
        self.chexpert_train_df_path = chexpert_train_df_path
        self.chexpert_pseudo_train_df_path = chexpert_pseudo_train_df_path

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predictions = torch.cat(predictions, dim=0)
        preds = predictions.float().squeeze(0).detach().cpu().numpy()

        np.save("chexpert_preds.npy", preds)
        
        chexpert_train_df = pd.read_csv(self.chexpert_train_df_path)
        org = np.array(chexpert_train_df.iloc[:, -26:].values).astype(np.float32)
        
        # Replace the original labels with the pseudo labels only if org value is -1
        idx = np.where(org == -1)
        org[idx] = preds[idx]

        chexpert_train_df.iloc[:, -26:] = org
        chexpert_train_df.to_csv(self.chexpert_pseudo_train_df_path, index=False)

        print(f"Chexpert pseudo labels saved to {self.chexpert_pseudo_train_df_path}")