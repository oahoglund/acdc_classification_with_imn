import lightning as L
import torch
import timm
import torch.nn as nn
import wandb

import numpy as np

from lightning.pytorch.loggers import WandbLogger

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
    MulticlassAveragePrecision,
    MulticlassConfusionMatrix,
)

from src.utils.logger import LocalImageLogger



class ClassificationBaseline(L.LightningModule):
    def __init__(self, encoder_name, in_channels, num_classes, criterion, lr, freeze_backbone_epochs=5, backbone_lr_mult=0.05,**kwargs):
        super().__init__()
        self.backbone = timm.create_model(encoder_name, pretrained=True, in_chans=in_channels, num_classes=0)
        nfeat = self.backbone.num_features
        self.head = nn.Linear(nfeat, num_classes)

        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.backbone_lr_mult = backbone_lr_mult

        self.metric_prefix = "val"

        self.criterion = criterion # loss function
        self.num_classes = num_classes
        self.lr = lr

        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1_per_class = MulticlassF1Score(num_classes=num_classes, average="none")

        self.val_precision_per_class = MulticlassPrecision(num_classes=num_classes, average="none")
        self.val_recall_per_class = MulticlassRecall(num_classes=num_classes, average="none")

        # # Optional (needs probs, i.e. softmax outputs)
        self.val_auroc = MulticlassAUROC(num_classes=num_classes, average="macro", thresholds=64)
        self.val_auprc = MulticlassAveragePrecision(num_classes=num_classes, average="macro", thresholds=64)

        self.val_conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.last_conf_matrix = None  # set after each validation epoch


    def forward(self, x):            # x: (B,36,H,W)
        feat = self.backbone(x)          # (B, nfeat)
        logits = self.head(feat)
        return logits
    
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def on_fit_start(self):
        self.freeze_backbone()

    def on_train_epoch_start(self):
        if self.current_epoch == self.freeze_backbone_epochs:
            self.unfreeze_backbone()

    def training_step(self, batch, batch_idx):
        ed_img, ed_gt, es_img, es_gt, info = batch
        y = info["group"]
        out = torch.cat([ed_img, es_img], dim=1)
        logits = self.forward(out)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        # probs = logits.softmax(dim=1)
        B = preds.size(0)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx):
        ed_img, ed_gt, es_img, es_gt, info = batch
        y = info["group"]
        out = torch.cat([ed_img, es_img], dim=1)
        logits = self.forward(out)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        probs = logits.softmax(dim=1)

        B = preds.size(0)

        self.val_acc.update(preds, y)
        self.val_f1_macro.update(preds, y)
        self.val_f1_per_class.update(preds, y)
        self.val_precision_per_class.update(preds, y)
        self.val_recall_per_class.update(preds, y)

        # optional:
        self.val_auroc.update(probs, y)
        self.val_auprc.update(probs, y)

        self.val_conf_matrix.update(preds, y)

        self.log(f"{self.metric_prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        return loss
    
    def on_validation_epoch_end(self):
        prefix = self.metric_prefix
        self.log(f"{prefix}/acc", self.val_acc.compute(), prog_bar=True)
        self.log(f"{prefix}/f1_macro", self.val_f1_macro.compute(), prog_bar=True)

        f1_pc = self.val_f1_per_class.compute()
        prec_pc = self.val_precision_per_class.compute()
        rec_pc = self.val_recall_per_class.compute()

        for i in range(self.num_classes):
            self.log(f"{prefix}/f1_c{i}", f1_pc[i])
            self.log(f"{prefix}/prec_c{i}", prec_pc[i])
            self.log(f"{prefix}/rec_c{i}", rec_pc[i])

        # # optional
        self.log(f"{prefix}/auroc_macro", self.val_auroc.compute())
        self.log(f"{prefix}/auprc_macro", self.val_auprc.compute())

        self.last_conf_matrix = self.val_conf_matrix.compute().cpu().numpy()

        # reset
        self.val_acc.reset()
        self.val_f1_macro.reset()
        self.val_f1_per_class.reset()
        self.val_precision_per_class.reset()
        self.val_recall_per_class.reset()
        self.val_auroc.reset()
        self.val_auprc.reset()
        self.val_conf_matrix.reset()
    
    def test_step(self, batch, batch_idx):
        pass


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.head.parameters(), "lr": self.lr},
                {"params": self.backbone.parameters(), "lr": self.lr * self.backbone_lr_mult},
            ],
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
