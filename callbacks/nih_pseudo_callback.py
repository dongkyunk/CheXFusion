import numpy as np
import torch
import pandas as pd
from lightning.pytorch.callbacks import BasePredictionWriter


class NihWriter(BasePredictionWriter):
    def __init__(self, nih_train_df_path, nih_pseudo_train_df_path, write_interval):
        super().__init__(write_interval)
        self.nih_train_df_path = nih_train_df_path
        self.nih_pseudo_train_df_path = nih_pseudo_train_df_path

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predictions = torch.cat(predictions, dim=0)
        preds = predictions.float().squeeze(0).detach().cpu().numpy()

        np.save("nih_preds.npy", preds)
        
        nih_train_df = pd.read_csv(self.nih_train_df_path)
        org = np.array(nih_train_df.iloc[:, -26:].values).astype(np.float32)
        
        # Replace the original labels with the pseudo labels only if org value is -1
        idx = np.where(org == -1)
        org[idx] = preds[idx]

        nih_train_df.iloc[:, -26:] = org
        nih_train_df.to_csv(self.nih_pseudo_train_df_path, index=False)

        print(f"NIH pseudo labels saved to {self.nih_pseudo_train_df_path}")