import torch
import lightning.pytorch as pl
from torch.optim import AdamW
from torchmetrics import AveragePrecision, AUROC
from transformers import get_cosine_schedule_with_warmup
from model.layers import Backbone, FusionBackbone
from model.loss import get_loss


class CxrModel(pl.LightningModule):
    def __init__(self, lr, classes, loss_init_args, timm_init_args):
        super(CxrModel, self).__init__()
        self.lr = lr
        self.classes = classes
        self.backbone = FusionBackbone(timm_init_args, 'model.pth')
        # self.backbone = Backbone(timm_init_args)
        self.validation_step_outputs = []
        self.val_ap = AveragePrecision(task='binary')
        self.val_auc = AUROC(task="binary")
        
        self.criterion_cls = get_loss(**loss_init_args)

    def forward(self, image):
        return self.backbone(image)

    def shared_step(self, batch, batch_idx):
        image, label = batch
        pred = self(image)

        loss = self.criterion_cls(pred, label)

        pred=torch.sigmoid(pred).detach()

        return dict(
            loss=loss,
            pred=pred,
            label=label,
        )

    def training_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self.log_dict({'loss': res['loss'].detach()}, prog_bar=True)
        self.log_dict({'train_loss': res['loss'].detach()}, prog_bar=True, on_step=False, on_epoch=True)
        return res['loss']
        
    def validation_step(self, batch, batch_idx):
        res = self.shared_step(batch, batch_idx)
        self.log_dict({'val_loss': res['loss'].detach()}, prog_bar=True)
        self.validation_step_outputs.append(res)

    def on_validation_epoch_end(self):
        preds = torch.cat([x['pred'] for x in self.validation_step_outputs])
        labels = torch.cat([x['label'] for x in self.validation_step_outputs])

        val_ap = []
        val_auroc = []
        for i in range(26):
            ap = self.val_ap(preds[:, i], labels[:, i].long())
            auroc = self.val_auc(preds[:, i], labels[:, i].long())
            val_ap.append(ap)
            val_auroc.append(auroc)
            print(f'{self.classes[i]}_ap: {ap}')
        
        head_idx = [0, 2, 4, 12, 14, 16, 20, 24]
        medium_idx = [1, 3, 5, 6, 8, 9, 10, 13, 15, 22]
        tail_idx = [7, 11, 17, 18, 19, 21, 23, 25]

        self.log_dict({'val_ap': sum(val_ap)/26}, prog_bar=True)
        self.log_dict({'val_auroc': sum(val_auroc)/26}, prog_bar=True)
        self.log_dict({'val_head_ap': sum([val_ap[i] for i in head_idx]) / len(head_idx)}, prog_bar=True)
        self.log_dict({'val_medium_ap': sum([val_ap[i] for i in medium_idx]) / len(medium_idx)}, prog_bar=True)
        self.log_dict({'val_tail_ap': sum([val_ap[i] for i in tail_idx]) / len(tail_idx)}, prog_bar=True)
        self.validation_step_outputs = []

    def predict_step(self, batch, batch_idx):
        pred = self.shared_step(batch, batch_idx)['pred']
        image, label = batch
        batch_1 = (image.flip(-1), label)
        pred_1 = self.shared_step(batch_1, batch_idx)['pred']
        pred = (pred + pred_1) / 2
        return pred

    def configure_optimizers(self):
        optimizer = AdamW(self.backbone.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 0, 250000)
        return [optimizer], [scheduler]