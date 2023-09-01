import torch
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import NeptuneLogger
from model.cxr_model import CxrModel
from dataset.cxr_datamodule import CxrDataModule

class MyLightningCLI(LightningCLI):
    def before_fit(self):
        if isinstance(self.trainer.logger, NeptuneLogger):
            self.trainer.logger.experiment["train/config"].upload('config.yaml')

def cli_main():
    torch.set_float32_matmul_precision('high')
    cli = MyLightningCLI(CxrModel, CxrDataModule, save_config_callback=None)

if __name__ == "__main__":
    cli_main()
