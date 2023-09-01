import torch
import yaml
from model.cxr_model import CxrModel

config = yaml.load(open('config.yaml', "r"), Loader=yaml.FullLoader)
model = CxrModel.load_from_checkpoint(
    config['ckpt_path'], **config['model']
)

torch.save(model.backbone.model.state_dict(), 'model.pth')
