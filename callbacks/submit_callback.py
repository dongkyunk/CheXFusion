import os
import zipfile
import torch
import pandas as pd
from lightning.pytorch.callbacks import BasePredictionWriter


class SubmissonWriter(BasePredictionWriter):
    def __init__(self, sample_submission_path, submission_path, submission_zip_path, submission_code_dir, pred_df_path, write_interval):
        super().__init__(write_interval)
        self.sample_submission_path = sample_submission_path
        self.submission_path = submission_path
        self.submission_zip_path = submission_zip_path
        self.submission_code_dir = submission_code_dir
        self.pred_df_path = pred_df_path

    def postprocess(self, submit_df):
        pred_df = pd.read_csv(self.pred_df_path)
        submit_df['study_id'] = pred_df['study_id']

        submit_df['weight'] = 5
        submit_df.loc[pred_df['ViewPosition'].isin(['PA', 'AP']), 'weight'] = 7
        submit_df.loc[pred_df['ViewPosition'].isin(['LL', 'LATERAL']), 'weight'] = 3

        # weighted average values across images and fix that value for all images in that study
        classes = submit_df.columns[1:-2]
        submit_df[classes] = submit_df[classes].mul(submit_df['weight'], axis=0)
        submit_df[classes] = submit_df.groupby('study_id')[classes].transform('sum') 
        submit_df['weight'] = submit_df.groupby('study_id')['weight'].transform('sum')
        submit_df[classes] = submit_df[classes].div(submit_df['weight'], axis=0)

        submit_df.drop(columns=['weight'], inplace=True)
        submit_df.drop(columns=['study_id'], inplace=True)

        return submit_df

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # Add predictions
        predictions = torch.cat(predictions, dim=0)
        # torch.save(predictions, "predictions.pt")

        submit_df = pd.read_csv(self.sample_submission_path)
        submit_df.iloc[..., 1:] = predictions.float().squeeze(
            0).detach().cpu().numpy()

        # Post processing
        submit_df = self.postprocess(submit_df)

        # Save submission
        submit_df.to_csv(self.submission_path, index=False)
        with zipfile.ZipFile(self.submission_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add the folder and its contents to the zip
            for root, _, files in os.walk(self.submission_code_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.join('code',os.path.relpath(file_path, self.submission_code_dir)))

            # Add the file to the zip
            zipf.write(self.submission_path, os.path.basename(self.submission_path))
        
        print(f"Submission saved!")
