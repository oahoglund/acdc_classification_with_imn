import lightning as L
import torch
import timm
import torch.nn as nn
import wandb

import numpy as np

from lightning.pytorch.loggers import WandbLogger

from src.utils.logger import LocalImageLogger

import segmentation_models_pytorch as smp

from torchmetrics.classification import MulticlassJaccardIndex, MulticlassF1Score, MulticlassConfusionMatrix

from src.utils.logger import LocalImageLogger


# This is the that takes just slices of the images stacks as input and predicts the corresponding mask slices, with some special handling for empty slices and patient-level logging in test.


class ACDCSegmentation2(L.LightningModule):
    def __init__(self, encoder_name, in_channels, mask_classes, criterion, lr, arch = "Unet", ignore_index = 255,**kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=["criterion"])

        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=mask_classes,
            **kwargs,
        )

        self.metric_prefix = "val"

        self.criterion = criterion # loss function
        self.num_classes = mask_classes
        self.lr = lr

        self.val_average_iou = MulticlassJaccardIndex(
            num_classes=mask_classes,
            average="micro",
            ignore_index=ignore_index,
        )

        self.val_average_dice = MulticlassF1Score(
            num_classes=mask_classes,
            average="micro",
            ignore_index=ignore_index,
        )

        self.val_pr_class_iou = MulticlassJaccardIndex(
            num_classes=mask_classes,
            average="none",
            ignore_index=ignore_index,
        )

        self.val_pr_class_dice = MulticlassF1Score(
            num_classes=mask_classes,
            average="none",
            ignore_index=ignore_index,
        )

        self.val_conf_matrix = MulticlassConfusionMatrix(num_classes=mask_classes, ignore_index=ignore_index)
        self.last_conf_matrix = None


    def forward(self, x):
        return self.model(x)
    

    def training_step(self, batch, batch_idx):
        img, mask, info = batch
        logits_mask = self.forward(img)
        loss = self.criterion(logits_mask, mask)
        preds = torch.argmax(logits_mask, dim=1)
        B = preds.size(0)

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B)
        return loss

    def validation_step(self, batch, batch_idx):
        img, mask, info = batch
        logits_mask = self.forward(img)
        loss = self.criterion(logits_mask, mask)
        preds = torch.argmax(logits_mask, dim=1)
        B = preds.size(0)
        self.log(f"{self.metric_prefix}/loss", loss, prog_bar=True, batch_size=B)

        self.val_average_dice.update(preds, mask)
        self.val_average_iou.update(preds, mask)
        self.val_pr_class_dice.update(preds, mask)
        self.val_pr_class_iou.update(preds, mask)
        self.val_conf_matrix.update(preds, mask)

        self.log(f"{self.metric_prefix}/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log(f"{self.metric_prefix}/dice_micro_mean", self.val_average_dice, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        self.log(f"{self.metric_prefix}/iou_micro_mean", self.val_average_iou, on_step=False, on_epoch=True, prog_bar=True, batch_size=B)
        return loss
    
    def on_validation_epoch_end(self):
        prefix = self.metric_prefix
        iou_per_class = self.val_pr_class_iou.compute()
        dice_per_class = self.val_pr_class_dice.compute()

        for i, (iou, dice) in enumerate(zip(iou_per_class, dice_per_class)):
            self.log(f"{prefix}/iou_class_{i}", iou, prog_bar=False)
            self.log(f"{prefix}/dice_class_{i}", dice, prog_bar=False)

        # all-class means (includes background)
        self.log(f"{prefix}/iou_macro_mean", iou_per_class.mean(), prog_bar=False)
        self.log(f"{prefix}/dice_macro_mean", dice_per_class.mean(), prog_bar=True)

        # foreground-only macro (exclude class 0 = background)
        self.log(f"{prefix}/dice_macro_fg", dice_per_class[1:].mean(), prog_bar=True)
        self.log(f"{prefix}/iou_macro_fg", iou_per_class[1:].mean(), prog_bar=True)

        # foreground micro: aggregate TP/FP/FN over classes 1..K-1 from confusion matrix
        cm = self.val_conf_matrix.compute()
        fg = list(range(1, self.num_classes))
        tp = sum(cm[c, c] for c in fg).float()
        fp = sum(cm[:, c].sum() - cm[c, c] for c in fg).float()
        fn = sum(cm[c, :].sum() - cm[c, c] for c in fg).float()
        self.log(f"{prefix}/dice_micro_fg", (2 * tp) / (2 * tp + fp + fn + 1e-6), prog_bar=False)
        self.log(f"{prefix}/iou_micro_fg", tp / (tp + fp + fn + 1e-6), prog_bar=False)

        self.last_conf_matrix = cm.cpu().numpy()

        self.val_pr_class_iou.reset()
        self.val_pr_class_dice.reset()
        self.val_average_iou.reset()
        self.val_average_dice.reset()
        self.val_conf_matrix.reset()

    @staticmethod
    def _select_apical_mid_basal(non_empty_idx: torch.Tensor):
        """
        non_empty_idx: 1D tensor of indices (sorted) for real slices.
        returns a python list of 3 indices [apical, mid, basal]
        """
        if non_empty_idx.numel() == 0:
            return []
        non_empty_idx = non_empty_idx.sort().values
        ap = non_empty_idx[0].item()
        ba = non_empty_idx[-1].item()
        mi = non_empty_idx[non_empty_idx.numel() // 2].item()
        return [ap, mi, ba]
    
    @staticmethod
    def _dice_macro_foreground(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6):
        """
        pred, target: (N, H, W) int tensors
        Computes mean Dice over foreground classes 1..K-1.
        If num_classes==1, falls back to binary-ish using class 1 not possible; so treat as class {0,1} if present.
        """
        pred = pred.long()
        target = target.long()

        if num_classes <= 1:
            # fallback: treat non-zero as foreground
            pred_fg = (pred != 0)
            tgt_fg  = (target != 0)
            inter = (pred_fg & tgt_fg).sum().float()
            denom = pred_fg.sum().float() + tgt_fg.sum().float()
            return (2.0 * inter + eps) / (denom + eps)

        dices = []
        for c in range(1, num_classes):
            pred_c = (pred == c)
            tgt_c  = (target == c)
            inter = (pred_c & tgt_c).sum().float()
            denom = pred_c.sum().float() + tgt_c.sum().float()
            dice_c = (2.0 * inter + eps) / (denom + eps)
            dices.append(dice_c)

        if len(dices) == 0:
            return torch.tensor(0.0, device=pred.device)
        return torch.stack(dices).mean()


    # -------------------------
    # Lightning hooks for testing
    # -------------------------

    def on_test_start(self):
        # patient_id -> dict with dice + slices to log later
        self._test_patient_results = {}


    def test_step(self, batch, batch_idx):
        ed_img, ed_gt, es_img, es_gt, info = batch
        # ed_img/es_img: (B, 18, H, W)
        # ed_gt /es_gt : (B, 18, H, W)
        # info: metadata, only use patient_id

        B, Z, H, W = ed_img.shape
        patient_ids = patient_ids = info["patient_id"]

        # concat ED+ES along "channel" dimension like you did in train/val
        x = torch.cat([ed_img, es_img], dim=1)  # (B, 36, H, W)
        y = torch.cat([ed_gt,  es_gt],  dim=1)  # (B, 36, H, W)

        B, C, H, W = x.shape  # C=36

        # flatten into (B*C, 1, H, W)
        x_flat = x.reshape(B * C, 1, H, W)
        y_flat = y.reshape(B * C, H, W).long()

        # non-empty keep mask (per slice)
        eps = 0.0
        keep = (x_flat.abs().sum(dim=(1, 2, 3)) > eps)  # (B*C,)

        # testing if this makes a difference
        keep = (y_flat != 0).any(dim=(1, 2))


        # if all empty, nothing to do
        if keep.sum() == 0:
            return None

        # forward only on non-empty
        x_nz = x_flat[keep]              # (N, 1, H, W)
        y_nz = y_flat[keep]              # (N, H, W)
        logits_nz = self.model(x_nz)     # (N, K, H, W)
        preds_nz = logits_nz.argmax(dim=1)  # (N, H, W)

        num_classes = logits_nz.shape[1]

        # put preds back into full (B*C, H, W) with a sentinel for empty slices
        preds_full = torch.full(
            (B * C, H, W),
            fill_value=-1,
            device=preds_nz.device,
            dtype=preds_nz.dtype
        )
        preds_full[keep] = preds_nz
        preds_full = preds_full.reshape(B, C, H, W)  # (B, 36, H, W)

        # also reshape GT to match (B, 36, H, W)
        y_full = y.reshape(B, C, H, W).long()

        # process each patient in the batch
        for b in range(B):
            pid = str(patient_ids[b])

            # all slices (ED+ES) non-empty for this patient
            x_b = x[b]          # (36, H, W)
            y_b = y_full[b]     # (36, H, W)
            p_b = preds_full[b] # (36, H, W), empty slices are -1

            # find valid slices for metric: pred != -1 (means slice was non-empty and inferred)
            valid = (p_b != -1)
            if valid.sum() == 0:
                dice_val = torch.tensor(0.0, device=p_b.device)
            else:
                pred_valid = p_b[valid]  # (Nvalid, H, W)
                tgt_valid  = y_b[valid]  # (Nvalid, H, W)
                dice_val = self._dice_macro_foreground(pred_valid, tgt_valid, num_classes=num_classes)

            # decide apical/mid/basal for ED and ES separately based on non-empty in the ORIGINAL ED/ES stacks
            # (padding is top/bottom; non-empty slices are the "real" ones)
            ed_non_empty = (ed_img[b].abs().sum(dim=(1, 2)) > eps)  # (18,)
            es_non_empty = (es_img[b].abs().sum(dim=(1, 2)) > eps)  # (18,)

            ed_idx = torch.where(ed_non_empty)[0]
            es_idx = torch.where(es_non_empty)[0]

            ed_sel = self._select_apical_mid_basal(ed_idx)  # indices in [0..17]
            es_sel = self._select_apical_mid_basal(es_idx)  # indices in [0..17]

            # stash everything needed for logging later
            # store CPU tensors to avoid holding GPU memory across epoch end
            def _pack_triplet(img_stack18, gt_stack18, pred_stack36, base_offset, sel3):
                # base_offset: 0 for ED, 18 for ES
                pack = []
                for z in sel3:
                    img = img_stack18[z].detach().float().cpu()   # (H, W)
                    gt  = gt_stack18[z].detach().long().cpu()     # (H, W)
                    pred = pred_stack36[base_offset + z].detach().long().cpu()  # (H, W) or -1
                    pack.append((z, img, gt, pred))
                return pack

            payload = {
                "dice": float(dice_val.detach().cpu().item()),
                "ed": _pack_triplet(ed_img[b], ed_gt[b], preds_full[b].detach().cpu(), 0,  ed_sel),
                "es": _pack_triplet(es_img[b], es_gt[b], preds_full[b].detach().cpu(), 18, es_sel),
            }

            # if a patient appears multiple times (shouldn't), keep the best record (or overwrite)
            self._test_patient_results[pid] = payload

        return None


    def on_test_epoch_end(self):
        # nothing to log
        if not hasattr(self, "_test_patient_results") or len(self._test_patient_results) == 0:
            return

        # sort patients by dice
        items = list(self._test_patient_results.items())  # [(pid, payload), ...]
        items_sorted = sorted(items, key=lambda kv: kv[1]["dice"])
        worst_pid, worst_payload = items_sorted[0]
        best_pid,  best_payload  = items_sorted[-1]
        med_pid,   med_payload   = items_sorted[len(items_sorted)//2]

        # log scalar summary too (optional)
        dices = torch.tensor([p["dice"] for _, p in items_sorted], dtype=torch.float32)
        self.log("test/dice_patient_macro_mean", dices.mean(), prog_bar=True)
        self.log("test/dice_patient_macro_median", dices.median(), prog_bar=True)
        self.log("test/dice_patient_best", torch.tensor(best_payload["dice"]), prog_bar=False)
        self.log("test/dice_patient_worst", torch.tensor(worst_payload["dice"]), prog_bar=False)
        
        def _to_uint8_numpy(x, is_mask=False):
            """
            x: torch.Tensor (H, W) or (1, H, W)
            is_mask: if True, assumes integer labels and does NOT normalize
            """
            if x.dim() == 3:
                x = x.squeeze(0)

            x = x.detach().cpu()

            if is_mask:
                return x.to(torch.uint8).numpy()

            # image: normalize to [0, 255]
            x = x.float()
            x = x - x.min()
            if x.max() > 0:
                x = x / x.max()
            x = (x * 255.0).clamp(0, 255)

            return x.to(torch.uint8).numpy()

        # helper to log triplets
        def _log_patient(name_prefix: str, pid: str, payload: dict):
            # payload["ed"] and payload["es"] are lists of (z, img(HW), gt(HW), pred(HW))
            for phase in ["ed", "es"]:
                triplets = payload[phase]
                for rank, (z, img, gt, pred) in zip(["apical", "mid", "basal"], triplets):
                    # make shapes consistent if your logger expects CHW
                    img_ = img.unsqueeze(0)  # (1, H, W)
                    # pred can be -1 if something went wrong; clamp to 0 for visualization if you want:
                    pred_ = pred.clone()
                    pred_[pred_ < 0] = 0

                    name = f"{pid}({name_prefix})_{phase}_{rank}"
                    for logger in self.loggers:
                        if isinstance(logger, LocalImageLogger):
                            logger.log_image(
                                name=name,
                                img_gray=_to_uint8_numpy(img_, is_mask=False),
                                gt=_to_uint8_numpy(gt, is_mask=True),
                                pred=_to_uint8_numpy(pred_, is_mask=True),
                            )

        _log_patient("worst",  worst_pid, worst_payload)
        _log_patient("median", med_pid,   med_payload)
        _log_patient("best",   best_pid,  best_payload)

        # (optional) free memory
        self._test_patient_results.clear()


    @staticmethod
    def undo_crop_to_full_torch(pred_crop_padded: torch.Tensor,
                                orig_h: int,
                                orig_w: int,
                                x1: int, y1: int, x2: int, y2: int,
                                pad_left: int, pad_top: int, pad_right: int, pad_bottom: int,
                                fill_value: float = 0.0) -> torch.Tensor:
        """
        pred_crop_padded: torch.Tensor with spatial dims in the last two positions:
            (..., H, W)  e.g. (H,W), (C,H,W), (B,C,H,W)
        Returns:
            (..., orig_h, orig_w) with the crop pasted back in.
        """
        t = pred_crop_padded

        # 1) remove padding (inverse of pad_if_needed)
        H, W = t.shape[-2], t.shape[-1]
        t_unpadded = t[..., pad_top:H - pad_bottom, pad_left:W - pad_right]

        # 'crop_coords': (0, 0, 256, 512), 'pad_params': {'pad_top': 182, 'pad_bottom': 182, 'pad_left': 6, 'pad_right': 6}}
        # 'crop_coords': (0, 49, 550, 99), 'pad_params': {'pad_top': 0, 'pad_bottom': 0, 'pad_left': 153, 'pad_right': 153}}
        # 'crop_coords': (102, 49, 142, 99), 'pad_params': None}
        # Optional safety check
        exp_h = y2 - y1
        exp_w = x2 - x1
        if t_unpadded.shape[-2] != exp_h or t_unpadded.shape[-1] != exp_w:
            raise ValueError(
                f"Unpadded pred shape {tuple(t_unpadded.shape[-2:])} "
                f"does not match expected crop size {(exp_h, exp_w)} "
                f"from coords {(x1,y1,x2,y2)} with pads {(pad_left,pad_top,pad_right,pad_bottom)}."
            )

        # 2) paste into full canvas
        out_shape = list(t_unpadded.shape)
        out_shape[-2] = orig_h
        out_shape[-1] = orig_w
        full = torch.full(out_shape, fill_value=fill_value, device=t.device, dtype=t.dtype)

        full[..., y1:y2, x1:x2] = t_unpadded
        return full
    
    def predict_step(self, batch, batch_idx):
        ed_img, es_img, info, m = batch
        B, Z, H, W = ed_img.shape

        keep_ed = (ed_img.abs().sum(dim=(2, 3)) > 0)  # (B, Z)
        keep_es = (es_img.abs().sum(dim=(2, 3)) > 0)  # (B, Z)

        x = torch.cat([ed_img, es_img], dim=1).reshape(B * 2 * Z, 1, H, W)
        keep = (x.abs().sum(dim=(1, 2, 3)) > 0)

        preds = torch.zeros((B * 2 * Z, H, W), device=x.device, dtype=torch.long)

        if keep.any():
            preds[keep] = self(x[keep]).argmax(dim=1)

        preds = preds.reshape(B, 2 * Z, H, W)

        preds_ed_full = []
        preds_es_full = []

        for i in range(B):
            oh = int(m["orig_h"][i].item())
            ow = int(m["orig_w"][i].item())
            x1 = int(m["x1"][i].item())
            y1 = int(m["y1"][i].item())
            x2 = int(m["x2"][i].item())
            y2 = int(m["y2"][i].item())
            pl = int(m["pad_left"][i].item())
            pt = int(m["pad_top"][i].item())
            pr = int(m["pad_right"][i].item())
            pb = int(m["pad_bottom"][i].item())

            pred_full_i = self.undo_crop_to_full_torch(
                preds[i], oh, ow, x1, y1, x2, y2, pl, pt, pr, pb, fill_value=0
            )  # (2Z, orig_h, orig_w)

            preds_ed_full.append(pred_full_i[:Z])
            preds_es_full.append(pred_full_i[Z:])

        return {
            "patient_id": info["patient_id"],
            "preds_ed": preds_ed_full,
            "preds_es": preds_es_full,
            "keep_ed": keep_ed,
            "keep_es": keep_es,
        }
    



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-4,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )

        # optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=self.lr,
        # )


        # scheduler = torch.optim.lr_scheduler.StepLR(
        #     optimizer,
        #     step_size=50,
        #     gamma=0.1,
        # )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
    