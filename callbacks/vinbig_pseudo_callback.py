import numpy as np
import torch
import pandas as pd
from lightning.pytorch.callbacks import BasePredictionWriter


class VinBigWriter(BasePredictionWriter):
    def __init__(self, vinbig_train_df_path, vinbig_pseudo_train_df_path, write_interval):
        super().__init__(write_interval)
        self.vinbig_train_df_path = vinbig_train_df_path
        self.vinbig_pseudo_train_df_path = vinbig_pseudo_train_df_path

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        predictions = torch.cat(predictions, dim=0)
        preds = predictions.float().squeeze(0).detach().cpu().numpy()

        vinbig_train_df = pd.read_csv(self.vinbig_train_df_path)
        org = np.array(vinbig_train_df.iloc[:, -26:].values).astype(np.float32)

        # Replace the original labels with the pseudo labels only if org value is -1
        idx = np.where(org == -1)
        org[idx] = preds[idx]

        # If both column nodule and mass is 1, then replace the one with lower pred with the pred value
        both_ones_indices = np.where((org[:, 13] == 1) & (org[:, 15] == 1))[0]

        for index in both_ones_indices:
            if preds[index, 13] < preds[index, 15]:
                org[index, 13] = preds[index, 13]
            else:
                org[index, 15] = preds[index, 15]

        vinbig_train_df.iloc[:, -26:] = org
        vinbig_train_df.to_csv(self.vinbig_pseudo_train_df_path, index=False)

        print(f"VinBig pseudo labels saved to {self.vinbig_pseudo_train_df_path}")

