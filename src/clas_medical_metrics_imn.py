import lightning as L
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
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

from src.utils.clinical_metrics import clinical_metrics_from_volumes_torch_batched, volumes_from_mask_torch_batched


class ResidualBlock(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, x):
        return x + self.net(x)


class IMNClassifier(nn.Module):
    """
    IMN for tabular classification:
      - backbone: deep MLP (residual)
      - head: outputs per-sample linear classifier params W(x), b(x)
      - prediction: logits = sum_m W(x)_c,m * x_m + b(x)_c
    """
    def __init__(self, num_features: int, num_classes: int, d_model=64, n_blocks=2):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes

        self.in_proj = nn.Linear(num_features, d_model)
        self.blocks = nn.Sequential(*[ResidualBlock(d_model, d_model) for _ in range(n_blocks)])

        # Per-sample params for a linear model in original feature space:
        # output shape = C*(M+1)
        self.out_head = nn.Linear(d_model, num_classes * (num_features + 1))

    def generate_params(self, x):
        h = F.gelu(self.in_proj(x))
        h = self.blocks(h)
        params = self.out_head(h)  # (B, C*(M+1))

        B = x.size(0)
        C = self.num_classes
        M = self.num_features

        params = params.view(B, C, M + 1)
        W = params[:, :, :M]   # (B,C,M)
        b = params[:, :, M]    # (B,C)
        return W, b

    def forward(self, x, return_explanations: bool = False):
        W, b = self.generate_params(x)
        logits = (W * x.unsqueeze(1)).sum(-1) + b  # (B,C)

        if return_explanations:
            return logits, W, b
        return logits

    def l1_penalty(self, x):
        W, _ = self.generate_params(x)
        return W.abs().mean()

class ClassificationMedicalMetricsIMN(L.LightningModule):
    def __init__(self, num_classes, criterion, lr, num_features=5, l1_lambda=1e-4,
                 d_model=64, n_blocks=2, feature_mean=None, feature_std=None, **kwargs):
        super().__init__()

        self.save_hyperparameters(ignore=["criterion"])
        self.l1_lambda = l1_lambda

        self.net = IMNClassifier(
            num_features=num_features,
            num_classes=num_classes,
            d_model=d_model,
            n_blocks=n_blocks,
        )

        
        self.metric_prefix = "val"

        self.criterion = criterion # loss function
        self.num_classes = num_classes
        self.lr = lr

        if feature_mean is None:
            feature_mean = torch.zeros(num_features)
        if feature_std is None:
            feature_std = torch.ones(num_features)

        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32))

        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_f1_per_class = MulticlassF1Score(num_classes=num_classes, average="none")

        self.val_precision_per_class = MulticlassPrecision(num_classes=num_classes, average="none")
        self.val_recall_per_class = MulticlassRecall(num_classes=num_classes, average="none")

        # # Optional (needs probs, i.e. softmax outputs)
        self.val_auroc = MulticlassAUROC(num_classes=num_classes, average="macro", thresholds=64)
        self.val_auprc = MulticlassAveragePrecision(num_classes=num_classes, average="macro", thresholds=64)

        self.val_conf_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        self.last_conf_matrix = None


    def forward(self, x, return_explanations: bool = False):  # features: (B, M)
        return self.net(x, return_explanations)
    
    def normalize_features(self, metrics):
        return (metrics - self.feature_mean) / (self.feature_std + 1e-8)
    
    def get_features(self, ed, es, info):
        height = info["height"]    # (B,)
        weight = info["weight"]    # (B,)
        spacing = info["spacing"]  # (B, 3)

        volumes_ed = volumes_from_mask_torch_batched(ed, spacing)  # (B, 3)
        volumes_es = volumes_from_mask_torch_batched(es, spacing)  # (B, 3)

        metrics = clinical_metrics_from_volumes_torch_batched(
            volumes_ed, volumes_es, height, weight,
            myocardial_density=1.05, BSA_formula="dubois"
        )  # (B, 5)
        return metrics
    
    def training_step(self, batch, batch_idx):
        ed_img, ed_gt, es_img, es_gt, info = batch
        y = info["group"]
        metrics = self.get_features(ed_gt, es_gt, info)
        metrics = self.normalize_features(metrics)

        logits = self.forward(metrics)
        loss = self.criterion(logits, y)

        l1 = self.net.l1_penalty(metrics)
        loss = loss + self.l1_lambda * l1

        preds = torch.argmax(logits, dim=1)
        # probs = logits.softmax(dim=1)
        B = preds.size(0)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx):
        ed_img, ed_gt, es_img, es_gt, info = batch
        y = info["group"]
        metrics = self.get_features(ed_gt, es_gt, info)
        metrics = self.normalize_features(metrics)
        logits = self.forward(metrics)
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

    def predict_step(self, batch, batch_idx):
        ed_img, ed_gt, es_img, es_gt, info = batch
        y = info["group"]

        metrics = self.get_features(ed_gt, es_gt, info)
        metrics = self.normalize_features(metrics)

        logits, W, b = self.forward(metrics, return_explanations=True)

        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        id = info["patient_id"]
        return {
            "patient_id": id.detach().cpu() if torch.is_tensor(id) else id,
            "preds": preds.detach().cpu(),
            "probs": probs.detach().cpu(),
            "targets": y.detach().cpu() if torch.is_tensor(y) else y,
            "features": metrics.detach().cpu(),
            "W": W.detach().cpu(),
            "b": b.detach().cpu(),
        }


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
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
