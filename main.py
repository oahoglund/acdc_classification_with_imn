import os
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "0"
os.environ["MIOPEN_LOG_LEVEL"] = "2"

SEED = 67

import torch
torch.set_float32_matmul_precision('highest')

from lightning.pytorch import seed_everything, Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

# Ensure reproducibility
seed_everything(SEED, workers=True)
torch.use_deterministic_algorithms(True)

from pathlib import Path
import numpy as np

from src.datamodules.acdc_dm import ACDCDataModule

from src.classificationbaseline import ClassificationBaseline
from src.clas_from_mask import ClassificationFromMask
from src.clas_medical_metrics_mlp import ClassificationMedicalMetricsMLP
from src.clas_medical_metrics_imn import ClassificationMedicalMetricsIMN


from src.acdc_segmentation2 import ACDCSegmentation2

from src.utils.logger import LocalImageLogger, get_next_run_dir, find_ckpts as _find_ckpts

from src.utils.transform import ACDC_Augmentations
from albumentations import CenterCrop

from src.utils.clinical_metrics import compute_feature_stats

from plot_conf_matrices import plot_conf_matrix as _plot_conf_matrix

import segmentation_models_pytorch as smp

import cv2
cv2.setNumThreads(0)

def no_cross():
    """
    simple function for testing model training withhout cross validation
    """
    model_name = "medical_metrics_imn"
    
    import wandb
    wandb.login()
    wandb_logger = WandbLogger(
        project="acdc_classification",
        name = f"{model_name}_no_cross_val"
    )
    run_dir = get_next_run_dir(Path("local_log"))
    # image_logger = LocalImageLogger(
    #     save_dir=run_dir / "images"
    # )

    in_channels = 18*2
    target_size = 256
    ts = ACDC_Augmentations(CenterCrop(target_size,target_size,pad_if_needed=True),target_z=in_channels//2)
    datamodule = ACDCDataModule(
        root=r"data\ACDC\database\training_npy",
        fold_idx=0,
        train_transform=ts.get_training_augmentation(),
        val_transform=ts.get_validation_augmentation(),
        batch_size=6,
        slice_mode=False,
    )
    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0,1.0], dtype=torch.float32) # 5 classes
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    model = get_model(model_name, criterion=criterion, lr=1e-3, num_classes=5, l1_lambda=1e-4)

    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=1e-3, patience=15, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val/acc", mode = "max", dirpath=run_dir, save_last=True, save_top_k=1, filename="best-{epoch:03d}-{val_acc:.4f}")

    trainer = Trainer(
        precision="32-true",
        accelerator="gpu",
        devices=1,
        enable_model_summary=False,
        logger = wandb_logger,
        max_epochs=100,
        log_every_n_steps=5,
        callbacks=[early_stop_callback,checkpoint_callback]
    )

    trainer.fit(model, datamodule=datamodule)

def cross_val(model_name: str, lr: float, extra_naming: str = "deterministic_final",
              save_info_path: Path = None, use_seg_masks: bool = False):
    """
    Function for running cross validation on the classification models. Works both with ground truth and predicted segmentation masks (controlled by use_seg_masks).
    It also logs everything to wandb and saves per fold and aggregated confusion matrices. 
    If save_info_path is provided, it will save the local run path of the first fold's run info json to that path for later reference.
    """
    import wandb
    import json
    wandb.login()

    suffix     = "_seg_mask" if use_seg_masks else ""
    class_names = ["DCM", "HCM", "MINF", "NOR", "ARV"]
    in_channels = 18*2
    target_size = 256
    ts = ACDC_Augmentations(CenterCrop(target_size, target_size, pad_if_needed=True), target_z=in_channels//2)

    class_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    fold_scores = []
    fold_conf_matrices = []

    for fold_idx in range(5):
        print(f"Starting fold {fold_idx}...")
        seed_everything(SEED + fold_idx, workers=True)

        # logging setup
        run_dir = get_next_run_dir(Path("local_log"))
        run_name = f"{model_name}_{extra_naming}{suffix}_fold_{fold_idx}"
        group    = f"{model_name}_{extra_naming}{suffix}_cross_val"
        wandb_logger = WandbLogger(project="acdc_classification", name=run_name, group=group)
        _save_run_info(run_dir, wandb_run_name=run_name, model_name=model_name,
                       extra_naming=extra_naming, fold_idx=fold_idx,
                       project="acdc_classification", group=group)

        if fold_idx == 0 and save_info_path is not None:
            save_info_path = Path(save_info_path)
            save_info_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_info_path, "w") as f:
                json.dump({"local_run_path": str(run_dir)}, f, indent=2)

        # other setup
        datamodule = ACDCDataModule(
            root=r"data\ACDC\database\training_npy",
            fold_idx=fold_idx,
            train_transform=ts.get_training_augmentation(),
            val_transform=ts.get_validation_augmentation(),
            batch_size=6,
            slice_mode=False,
            use_fold_masks=use_seg_masks,
        )

        if model_name.startswith("medical_metrics"):
            datamodule.setup(stage="fit")
            feature_mean, feature_std = compute_feature_stats(datamodule.train_dataloader())
        else:
            feature_mean = None
            feature_std = None

        model = get_model(model_name, criterion=criterion, lr=lr, num_classes=5,
                          feature_mean=feature_mean, feature_std=feature_std)

        checkpoint_callback = ModelCheckpoint(monitor="val/acc", mode="max", dirpath=run_dir,
                                              save_last=True, save_top_k=1, filename="best-{epoch:03d}")
        early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=1e-4, patience=25,
                                            verbose=False, mode="min")

        trainer = Trainer(
            precision="32-true",
            accelerator="gpu",
            devices=1,
            enable_model_summary=False,
            logger=wandb_logger,
            max_epochs=200,
            log_every_n_steps=5,
            callbacks=[early_stop_callback, checkpoint_callback],
        )

        # training
        trainer.fit(model, datamodule=datamodule)

        # get metrics
        best_path = checkpoint_callback.best_model_path
        val_metrics = trainer.validate(model=None, datamodule=datamodule, ckpt_path=best_path)[0]
        fold_scores.append({k: float(v) for k, v in val_metrics.items()})

        cm = model.last_conf_matrix
        y_true, y_pred = [], []
        for ti in range(len(class_names)):
            for pj in range(len(class_names)):
                count = int(cm[ti, pj])
                y_true.extend([ti] * count)
                y_pred.extend([pj] * count)
        wandb_logger.experiment.log({
            f"fold_{fold_idx}/confusion_matrix": wandb.plot.confusion_matrix(
                y_true=y_true, preds=y_pred, class_names=class_names
            )
        })
        fold_conf_matrices.append(cm)
        wandb.finish()

    # summary logging
    run = wandb.init(project="acdc_classification",
                     name=f"{model_name}_{extra_naming}{suffix}_cv_summary",
                     group=f"{model_name}_{extra_naming}{suffix}_cross_val")

    metric_names = sorted(fold_scores[0].keys())
    for metric_name in metric_names:
        values = np.array([fold[metric_name] for fold in fold_scores], dtype=float)
        safe_name = metric_name.replace("/", "_")
        run.summary[f"cv_mean_{safe_name}"] = float(values.mean())
        run.summary[f"cv_std_{safe_name}"] = float(values.std(ddof=1))

    agg_cm = np.sum(fold_conf_matrices, axis=0)
    y_true, y_pred = [], []
    for ti in range(len(class_names)):
        for pj in range(len(class_names)):
            count = int(agg_cm[ti, pj])
            y_true.extend([ti] * count)
            y_pred.extend([pj] * count)
    run.log({
        "cv_summary/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_true, preds=y_pred, class_names=class_names
        )
    })
    run.finish()

def get_model(name: str, criterion: torch.nn.Module, lr: float = 1e-3, in_channels: int = 18*2, num_classes: int = 5, **kwargs) -> torch.nn.Module:
    """
    Factory function to create a classification model based on the specified name and parameters.

    Parameters
    ----------
    name : str
        Name of the model to create. One of "from_mask", "baseline", "medical_metrics_mlp", "medical_metrics_imn".
    criterion : torch.nn.Module
        Loss function to use for training.
    lr : float, optional
        Learning rate for the optimizer, by default 1e-3.
    in_channels : int, optional
        Number of input channels for the model, by default 18*2 (18 slices with 2 channels each).
    num_classes : int, optional
        Number of output classes for classification, by default 5 (DCM, HCM, MINF, NOR, ARV).
    **kwargs
        Additional keyword arguments specific to the model type. For "from_mask", can include "n_mask_classes" (default 4) and "encoder_name" (default "resnet34"). For "medical_metrics_imn", can include "l1_lambda" (default 1e-4).

    Returns
    -------
    model : torch.nn.Module
        The created model.
    """
    if name == "from_mask":
        n_mask_classes = kwargs.pop("n_mask_classes", 4)
        encoder_name = kwargs.pop("encoder_name", "resnet34") 
        model = ClassificationFromMask(encoder_name, in_channels=in_channels, num_classes=num_classes, n_mask_classes=n_mask_classes, criterion=criterion, lr=lr)
    elif name == "baseline":
        encoder_name = kwargs.pop("encoder_name", "resnet34")
        model = ClassificationBaseline(encoder_name, in_channels=in_channels, num_classes=num_classes, criterion=criterion, lr=lr)
    elif name == "medical_metrics_mlp":
        feature_mean = kwargs.pop("feature_mean", None)
        feature_std = kwargs.pop("feature_std", None)
        model = ClassificationMedicalMetricsMLP(
            num_classes=num_classes, criterion=criterion, lr=lr,
            feature_mean=feature_mean, feature_std=feature_std
            )
    elif name == "medical_metrics_imn":
        l1_lambda = kwargs.pop("l1_lambda", 1e-4)
        feature_mean = kwargs.pop("feature_mean", None)
        feature_std = kwargs.pop("feature_std", None)
        model = ClassificationMedicalMetricsIMN(
            num_classes=num_classes, criterion=criterion, lr=lr, l1_lambda=l1_lambda,
            feature_mean=feature_mean, feature_std=feature_std
        )
    else:
        raise ValueError(f"Unknown model name: {name}")
    
    # if kwargs: # If there are any unused kwargs, raise an error to alert the user
    #     raise TypeError(
    #         f"Unexpected arguments for model '{name}': {list(kwargs.keys())}"
    #     )
    
    return model


def _reverse_center_crop(vol: np.ndarray, H_orig: int, W_orig: int, target: int = 256) -> np.ndarray:
    """
    Reverse CenterCrop(target, target, pad_if_needed=True) applied to (Z, target, target) predictions,
    returning (Z, H_orig, W_orig) masks at the original image resolution.

    If dim >= target: the crop took the center `target` rows/cols -> embed them back.
    If dim < target:  the image was padded to `target` -> strip the padding back out.
    """
    # Y (height)
    if H_orig >= target:
        y0, y1 = 0, target
        out_y0 = (H_orig - target) // 2
        out_y1 = out_y0 + target
    else:
        pad_top = (target - H_orig) // 2
        y0, y1 = pad_top, pad_top + H_orig
        out_y0, out_y1 = 0, H_orig

    # X (width)
    if W_orig >= target:
        x0, x1 = 0, target
        out_x0 = (W_orig - target) // 2
        out_x1 = out_x0 + target
    else:
        pad_left = (target - W_orig) // 2
        x0, x1 = pad_left, pad_left + W_orig
        out_x0, out_x1 = 0, W_orig

    out = np.zeros((vol.shape[0], H_orig, W_orig), dtype=vol.dtype)
    out[:, out_y0:out_y1, out_x0:out_x1] = vol[:, y0:y1, x0:x1]
    return out

def _save_run_info(run_dir: Path, wandb_run_name: str, **kwargs):
    """Write a run_info.json into run_dir so checkpoints can be identified later."""
    import json
    info = {"wandb_run_name": wandb_run_name, **kwargs}
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_info.json", "w") as f:
        json.dump(info, f, indent=2)

def _save_conf_matrices_json(fold_conf_matrices: list, agg_cm: np.ndarray, class_names: list, run, artifact_name: str, save_dir: Path = None):
    """Upload per-fold and aggregated confusion matrices as a wandb artifact"""
    import json, tempfile, os
    import wandb
    data = {
        "class_names": class_names,
        "folds": [cm.tolist() for cm in fold_conf_matrices],
        "aggregated": agg_cm.tolist(),
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f, indent=2)
        tmp_path = f.name
    artifact = wandb.Artifact(artifact_name, type="confusion_matrix")
    artifact.add_file(tmp_path, name="conf_matrices.json")
    run.log_artifact(artifact)
    os.remove(tmp_path)


def segmenter2():
    """
    simple function for testing model training withhout cross validation
    """
    arch = "Unet"
    encoder = "resnet34"
    model_name = f"{arch.lower()}_{encoder}"
    
    import wandb
    wandb.login()
    wandb_logger = WandbLogger(
        project="acdc_segmentation",
        name = f"{model_name}_no_cross_val_ce_seg1_lr1e-2_weightdecay1e-4_cosineanealing_fold1_dataloader2"
    )
    run_dir = get_next_run_dir(Path("local_log"))
    image_logger = LocalImageLogger(
        save_dir=run_dir / "images"
    )

    in_channels = 1 # per slice
    target_size = 256
    ts = ACDC_Augmentations(CenterCrop(target_size,target_size,pad_if_needed=True),target_z=18)
    datamodule = ACDCDataModule(
        root=r"data\ACDC\database\training_npy",
        fold_idx=1,
        train_transform=ts.get_segmentation_training_augmentation_2d(),
        val_transform=ts.get_validation_augmentation_2d(),
        batch_size=60,
        slice_mode=True,
    )
    class_weights = torch.tensor([0.1, 0.3, 0.3, 0.3], dtype=torch.float32) # 4 classes
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
    #criterion = CE_Dice_Loss(class_weights=class_weights, ignore_index=255, dice_weight=1.0)
    #criterion = smp.losses.DiceLoss(mode="multiclass", from_logits=True, ignore_index=255)
    model = ACDCSegmentation2(encoder_name=encoder, in_channels=in_channels, mask_classes=4, criterion=criterion, lr=1e-2, arch=arch)

    early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=1e-3, patience=15, verbose=False, mode="min")
    checkpoint_callback = ModelCheckpoint(monitor="val/dice_macro_mean", mode = "max", dirpath=run_dir, save_last=True, save_top_k=1, filename="best-{epoch:03d}-{val_dice_macro_mean:.4f}")

    trainer = Trainer(
        precision="32-true",
        accelerator="gpu",
        devices=1,
        enable_model_summary=False,
        logger = [wandb_logger, image_logger],
        max_epochs=1,
        log_every_n_steps=5,
        callbacks=[early_stop_callback,checkpoint_callback]
    )

    trainer.fit(model, datamodule=datamodule)

    datamodule = ACDCDataModule(
        root=r"data\ACDC\database\training_npy",
        fold_idx=1,
        train_transform=ts.get_segmentation_training_augmentation(),
        val_transform=ts.get_validation_augmentation(),
        batch_size=4,
        slice_mode=False,
    )
    trainer.test(model,datamodule)

def cross_segmenter2(save_info_path: Path = None):
    """
    Function for running cross validation on the ACDCSegmentation2 model and it also saves per patient statistics
    """
    arch = "Unet"
    encoder = "resnet34"
    model_name = f"{arch.lower()}_{encoder}"
    extra_naming = "dice_lr1e-4_cosineanealing_dataloader2_patience40"
    
    import wandb
    wandb.login()

    in_channels = 1 # per slice
    target_size = 256
    ts = ACDC_Augmentations(CenterCrop(target_size,target_size,pad_if_needed=True),target_z=18)
    criterion = smp.losses.DiceLoss(mode="multiclass", from_logits=True, ignore_index=255)

    fold_scores = []
    fold_conf_matrices = []
    seg_class_names = ["BG", "RVC", "Myo", "LVC"]

    for fold_idx in range(5): # there are 5 fold pre made in datamodule
        print(f"Starting fold {fold_idx}...")
        seed_everything(SEED + fold_idx, workers=True)
        
        # logger setup
        run_dir = get_next_run_dir(Path("local_log"))

        wandb_logger = WandbLogger(
            project="acdc_segmentation",
            name = f"{model_name}_{extra_naming}_fold_{fold_idx}",
            group = f"{model_name}_{extra_naming}_cross_val"
        )
        run_dir = get_next_run_dir(Path("local_log"))
        _save_run_info(run_dir, wandb_run_name=f"{model_name}_{extra_naming}_fold_{fold_idx}",
                       model_name=model_name, extra_naming=extra_naming, fold_idx=fold_idx,
                       project="acdc_segmentation", group=f"{model_name}_{extra_naming}_cross_val")

        if fold_idx == 0 and save_info_path is not None:
            import json
            save_info_path = Path(save_info_path)
            save_info_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_info_path, "w") as f:
                json.dump({"local_run_path": str(run_dir)}, f, indent=2)

        image_logger = LocalImageLogger(
            save_dir=run_dir / "images"
        )

        # other setup
        datamodule = ACDCDataModule(
            root=r"data\ACDC\database\training_npy",
            fold_idx=fold_idx,
            train_transform=ts.get_segmentation_training_augmentation_2d(),
            val_transform=ts.get_validation_augmentation_2d(),
            batch_size=60,
            slice_mode=True,
        )

        model = ACDCSegmentation2(encoder_name=encoder, in_channels=in_channels, mask_classes=4, criterion=criterion, lr=1e-3, arch=arch)

        early_stop_callback = EarlyStopping(monitor="val/loss", min_delta=1e-4, patience=40, verbose=False, mode="min")
        checkpoint_callback = ModelCheckpoint(monitor="val/dice_macro_mean", mode = "max", dirpath=run_dir, save_last=True, save_top_k=1, filename="best-{epoch:03d}")

        trainer = Trainer(
            precision="32-true",
            accelerator="gpu",
            devices=1,
            enable_model_summary=False,
            logger = [wandb_logger, image_logger],
            max_epochs=200,
            log_every_n_steps=5,
            callbacks=[early_stop_callback,checkpoint_callback]
        )
        
        # training
        trainer.fit(model, datamodule=datamodule)

        # get metrics
        best_path = checkpoint_callback.best_model_path
        val_metrics = trainer.validate(model=None, datamodule=datamodule, ckpt_path=best_path)[0]
        clean_metrics = {k: float(v) for k, v in val_metrics.items()}
        fold_scores.append(clean_metrics)

        # log per-fold confusion matrix as normalized heatmap
        cm = model.last_conf_matrix
        fold_conf_matrices.append(cm)
        fig = _plot_conf_matrix(cm, seg_class_names)
        wandb_logger.experiment.log({f"fold_{fold_idx}/confusion_matrix": wandb.Image(fig)})
        import matplotlib.pyplot as plt
        plt.close(fig)

        datamodule = ACDCDataModule(
            root=r"data\ACDC\database\training_npy",
            fold_idx=fold_idx,
            train_transform=ts.get_segmentation_training_augmentation(),
            val_transform=ts.get_validation_augmentation(),
            batch_size=4,
            slice_mode=False,
        )
        trainer.test(model,datamodule)

        wandb.finish()

    # summary logging
    run = wandb.init(project="acdc_segmentation", name=f"{model_name}_{extra_naming}_cv_summary", group=f"{model_name}_{extra_naming}_cross_val")

    metric_names = sorted(fold_scores[0].keys())
    for metric_name in metric_names:
        values = np.array([fold[metric_name] for fold in fold_scores], dtype=float)

        # mean/std across folds
        safe_name = metric_name.replace("/", "_")
        run.summary[f"cv_mean_{safe_name}"] = float(values.mean())
        run.summary[f"cv_std_{safe_name}"] = float(values.std(ddof=1))

    # log aggregated confusion matrix (sum across all folds)
    import matplotlib.pyplot as plt
    save_dir = Path(r"results\cross_val\segmentation\unet")
    agg_cm = np.sum(fold_conf_matrices, axis=0)
    fig = _plot_conf_matrix(agg_cm, seg_class_names, save_path=save_dir / "conf_matrix_cv_summary.pdf")
    run.log({"cv_summary/confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

    _save_conf_matrices_json(fold_conf_matrices, agg_cm, seg_class_names,
                             run, f"{model_name}_{extra_naming}_conf_matrices")

    run.finish()


def test_ensemble_segmenter():
    """
    Ensemble segmentation on the held-out test set (patients 101-150).
    - Averages softmax from all 5 fold checkpoints per slice
    - Computes dice/iou/confusion matrix against ground truth
    - Saves ensemble predicted masks to data/ACDC/database/testing_cross_val/
      as ED_seg_ensemble.npy / ES_seg_ensemble.npy (for downstream classification)
    - Logs metrics + confusion matrix PDF/JSON to wandb and results folder
    """
    import wandb
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from torch.utils.data import DataLoader
    from src.datasets.acdc_dataset_slices import ACDC_SliceDataset as SliceDataset
    wandb.login()

    ckpts = _find_ckpts(Path(r"results\cross_val\segmentation\unet\info.json"))

    model_name = "unet_resnet34"
    extra_naming = "dice_lr1e-4_cosineanealing_dataloader2_patience40"
    seg_class_names = ["BG", "RVC", "Myo", "LVC"]
    target_size = 256
    test_root = Path(r"data\ACDC\database\testing_npy")
    save_root = Path(r"data\ACDC\database\testing_cross_val")
    results_dir = Path(r"results\test_set\segmentation\unet")

    # load test patients
    test_patients = sorted([p.name for p in test_root.iterdir() if p.is_dir()])
    print(f"Found {len(test_patients)} test patients")


    ts = ACDC_Augmentations(CenterCrop(target_size, target_size, pad_if_needed=True), target_z=18)
    test_ds = SliceDataset(test_root, patient_list=test_patients, augmentation=ts.get_validation_augmentation_2d())
    test_dl = DataLoader(test_ds, batch_size=60, shuffle=False, num_workers=4,
                         pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # load the models for the folds
    device = torch.device("cuda")
    criterion = smp.losses.DiceLoss(mode="multiclass", from_logits=True, ignore_index=255)
    models = [ACDCSegmentation2.load_from_checkpoint(ckpt, criterion=criterion).to(device).eval()
              for ckpt in ckpts]
    print("Loaded all 5 checkpoints")

    from torchmetrics.classification import MulticlassF1Score, MulticlassJaccardIndex, MulticlassConfusionMatrix
    conf_matrix  = MulticlassConfusionMatrix(num_classes=4, ignore_index=255).to(device)
    dice_pc      = MulticlassF1Score(num_classes=4, average="none",  ignore_index=255).to(device)
    iou_pc       = MulticlassJaccardIndex(num_classes=4, average="none",  ignore_index=255).to(device)
    dice_micro   = MulticlassF1Score(num_classes=4, average="micro", ignore_index=255).to(device)
    iou_micro    = MulticlassJaccardIndex(num_classes=4, average="micro", ignore_index=255).to(device)

    # collect per patient: (z, pred_np, gt_np, img_np, is_nonzero)
    patient_data = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for batch in test_dl:
            img, mask, info = batch
            img  = img.to(device)
            mask = mask.to(device)

            probs = torch.stack([m(img).softmax(dim=1) for m in models]).mean(dim=0)
            preds = probs.argmax(dim=1)

            conf_matrix.update(preds, mask)
            dice_pc.update(preds, mask)
            iou_pc.update(preds, mask)
            dice_micro.update(preds, mask)
            iou_micro.update(preds, mask)

            B = preds.size(0)
            for b in range(B):
                pid        = info["patient_id"][b]
                phase      = info["phase"][b]
                z          = info["slice_idx"][b].item()
                is_nonzero = bool(img[b].abs().sum().item() > 0)
                patient_data[pid][phase].append((
                    z,
                    preds[b].cpu().numpy(),
                    mask[b].cpu().numpy(),
                    img[b, 0].cpu().numpy(),   # (H,W) single channel
                    is_nonzero,
                ))

    # save ensemble masks at original resolution
    for pid, phases in patient_data.items():
        patient_dir = save_root / pid
        patient_dir.mkdir(parents=True, exist_ok=True)
        for phase, slices in phases.items():
            slices_sorted  = sorted(slices, key=lambda x: x[0])
            nonzero_preds  = [pred for _, pred, _, _, nz in slices_sorted if nz]
            if nonzero_preds:
                orig_img   = np.load(test_root / pid / f"{phase}_img.npy")
                H_orig, W_orig = orig_img.shape[1], orig_img.shape[2]
                vol_256 = np.stack(nonzero_preds)
                vol = _reverse_center_crop(vol_256, H_orig, W_orig, target=target_size)
                np.save(patient_dir / f"{phase}_seg_ensemble.npy", vol)
    print(f"Ensemble masks saved to {save_root}")

    # ── per-patient dice for best/median/worst selection ──────────────────
    from src.acdc_segmentation2 import ACDCSegmentation2 as _Seg2
    patient_dice = {}
    for pid, phases in patient_data.items():
        all_preds, all_gts = [], []
        for phase, slices in phases.items():
            for _, pred, gt, _, nz in slices:
                if nz:
                    all_preds.append(pred)
                    all_gts.append(gt)
        if all_preds:
            pred_t = torch.tensor(np.stack(all_preds), device=device)
            gt_t   = torch.tensor(np.stack(all_gts),   device=device)
            patient_dice[pid] = _Seg2._dice_macro_foreground(pred_t, gt_t, num_classes=4).item()

    items_sorted  = sorted(patient_dice.items(), key=lambda kv: kv[1])
    worst_pid = items_sorted[0][0]
    med_pid   = items_sorted[len(items_sorted) // 2][0]
    best_pid  = items_sorted[-1][0]

    # ── image logging (best / median / worst) ─────────────────────────────
    from src.utils.logger import LocalImageLogger
    image_logger = LocalImageLogger(save_dir=results_dir / "images")

    def _to_uint8(arr):
        arr = arr.astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        return (arr * 255).clip(0, 255).astype(np.uint8)

    for label, pid in [("worst", worst_pid), ("median", med_pid), ("best", best_pid)]:
        for phase, slices in patient_data[pid].items():
            slices_sorted = sorted(slices, key=lambda x: x[0])
            nonzero = [(z, pred, gt, img_s) for z, pred, gt, img_s, nz in slices_sorted if nz]
            if not nonzero:
                continue
            selected = [nonzero[0], nonzero[len(nonzero) // 2], nonzero[-1]]
            for rank, (z, pred, gt, img_s) in zip(["apical", "mid", "basal"], selected):
                image_logger.log_image(
                    name=f"{pid}({label})_{phase}_{rank}",
                    img_gray=_to_uint8(img_s),
                    gt=gt.astype(np.uint8),
                    pred=pred.astype(np.uint8),
                )

    # ── compute aggregate metrics ──────────────────────────────────────────
    cm           = conf_matrix.compute().cpu().numpy()
    dice_pc_vals = dice_pc.compute()
    iou_pc_vals  = iou_pc.compute()

    dice_macro_fg  = dice_pc_vals[1:].mean().item()
    iou_macro_fg   = iou_pc_vals[1:].mean().item()
    dice_macro_all = dice_pc_vals.mean().item()
    iou_macro_all  = iou_pc_vals.mean().item()

    cm_t = torch.tensor(cm, dtype=torch.float32)
    fg = [1, 2, 3]
    tp = sum(cm_t[c, c] for c in fg)
    fp = sum(cm_t[:, c].sum() - cm_t[c, c] for c in fg)
    fn = sum(cm_t[c, :].sum() - cm_t[c, c] for c in fg)
    dice_micro_fg_val = (2 * tp / (2 * tp + fp + fn + 1e-6)).item()
    iou_micro_fg_val  = (tp / (tp + fp + fn + 1e-6)).item()

    patient_dice_t = torch.tensor(list(patient_dice.values()))

    # log to wandb
    run = wandb.init(
        project="acdc_segmentation",
        name=f"{model_name}_{extra_naming}_test_ensemble",
        group=f"{model_name}_{extra_naming}_cross_val",
    )
    metrics = {
        "test/dice_micro_mean":        dice_micro.compute().item(),
        "test/iou_micro_mean":         iou_micro.compute().item(),
        "test/dice_macro_mean":        dice_macro_all,
        "test/iou_macro_mean":         iou_macro_all,
        "test/dice_macro_fg":          dice_macro_fg,
        "test/iou_macro_fg":           iou_macro_fg,
        "test/dice_micro_fg":          dice_micro_fg_val,
        "test/iou_micro_fg":           iou_micro_fg_val,
        "test/dice_patient_macro_mean":   patient_dice_t.mean().item(),
        "test/dice_patient_macro_median": patient_dice_t.median().item(),
        "test/dice_patient_best":         patient_dice_t.max().item(),
        "test/dice_patient_worst":        patient_dice_t.min().item(),
    }
    for i, name in enumerate(seg_class_names):
        metrics[f"test/dice_class_{i}"] = dice_pc_vals[i].item()
        metrics[f"test/iou_class_{i}"]  = iou_pc_vals[i].item()
        metrics[f"test/dice_{name}"]    = dice_pc_vals[i].item()
        metrics[f"test/iou_{name}"]     = iou_pc_vals[i].item()
    run.log(metrics)

    fig = _plot_conf_matrix(cm, seg_class_names, save_path=results_dir / "conf_matrix_test_ensemble.pdf")
    run.log({"test/confusion_matrix": wandb.Image(fig)})
    plt.close(fig)

    _save_conf_matrices_json([cm], cm, seg_class_names,
                             run, f"{model_name}_{extra_naming}_test_ensemble_conf_matrix")
    run.finish()

def test_ensemble_classifier(model_name: str, seg_variant: str):
    """
    Ensemble classification on the held-out test set (patients 101-150).

    seg_variant: "gt"  -> uses checkpoints from cross_val()        (trained on GT masks)
                 "seg" -> uses checkpoints from cross_val(..., use_seg_masks=True)

    For all mask-based models the ensemble segmentation masks saved by
    test_ensemble_segmenter() are loaded from disk — the UNet does NOT run here.

    Requires test_ensemble_segmenter() to have been run first for mask-based models.
    """
    import wandb
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    from src.datasets.acdc_dataset_patients import ACDC_PatientDataset as PatientDataset
    wandb.login()

    results_base = Path("results/cross_val/classification")
    uses_masks = model_name in ("from_mask", "medical_metrics_mlp", "medical_metrics_imn")

    # resolve short folder name used in results/
    _results_folder = {"baseline": "baseline", "from_mask": "from_mask",
                       "medical_metrics_mlp": "mlp", "medical_metrics_imn": "imn"}[model_name]

    if seg_variant == "gt":
        info_path = results_base / _results_folder / "info.json"
    else:
        info_path = results_base / _results_folder / "info_seg.json"

    ckpts = _find_ckpts(info_path)
    print(f"[{model_name}/{seg_variant}] Found checkpoints: {ckpts}")

    CLASS_NAMES = ["DCM", "HCM", "MINF", "NOR", "ARV"]
    target_size = 256
    test_npy_root = Path(r"data\ACDC\database\testing_npy")
    test_seg_root = Path(r"data\ACDC\database\testing_cross_val")
    results_dir   = Path(r"results\test_set\classification") / f"{model_name}_{seg_variant}"

    test_patients = sorted([p.name for p in test_npy_root.iterdir() if p.is_dir()])
    ts = ACDC_Augmentations(CenterCrop(target_size, target_size, pad_if_needed=True), target_z=18)

    # dataset: mask-based models load pre-saved ensemble masks from testing_cross_val/
    # no UNet runs here — masks are already on disk from test_ensemble_segmenter()
    if uses_masks:
        test_ds = PatientDataset(
            root=test_npy_root,
            patient_list=test_patients,
            augmentation=ts.get_validation_augmentation(),
            use_fold_masks=True,
            seg_root_override=test_seg_root,
            seg_suffix="ensemble",
        )
    else:
        test_ds = PatientDataset(
            root=test_npy_root,
            patient_list=test_patients,
            augmentation=ts.get_validation_augmentation(),
        )

    test_dl = DataLoader(test_ds, batch_size=6, shuffle=False, num_workers=4,
                         pin_memory=True, persistent_workers=True, prefetch_factor=2)

    device = torch.device("cuda")
    criterion = torch.nn.CrossEntropyLoss()

    # load 5 fold models from checkpoint — feature_mean/std are restored from ckpt for metric models
    # Load models — each class has different requirements:
    # - baseline/from_mask: no save_hyperparameters -> use get_model() + load_state_dict
    # - mlp: no save_hyperparameters, but feature_mean/std are buffers restored by load_state_dict;
    #        pass dummy tensors to __init__ so register_buffer doesn't fail, then overwrite
    # - imn: has save_hyperparameters(ignore=["criterion"]) -> load_from_checkpoint works
    dummy_stats = torch.zeros(5)  # overwritten by load_state_dict for metric models
    def _load_one(ckpt_path):
        if model_name == "medical_metrics_imn":
            m = ClassificationMedicalMetricsIMN.load_from_checkpoint(ckpt_path, criterion=criterion, strict=False)
        else:
            m = get_model(model_name, criterion=criterion, lr=1e-3, num_classes=5,
                          feature_mean=dummy_stats, feature_std=dummy_stats)
            state = torch.load(ckpt_path, map_location=device, weights_only=True)
            m.load_state_dict(state["state_dict"], strict=False)
        return m.to(device).eval()

    models = [_load_one(ckpt) for ckpt in ckpts]

    # metrics — match cross-val validation_step logging exactly
    from torchmetrics.classification import (
        MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall,
        MulticlassAUROC, MulticlassAveragePrecision, MulticlassConfusionMatrix,
    )
    acc_metric  = MulticlassAccuracy(num_classes=5).to(device)
    f1_metric   = MulticlassF1Score(num_classes=5, average="macro").to(device)
    f1_pc       = MulticlassF1Score(num_classes=5, average="none").to(device)
    prec_pc     = MulticlassPrecision(num_classes=5, average="none").to(device)
    rec_pc      = MulticlassRecall(num_classes=5, average="none").to(device)
    auroc       = MulticlassAUROC(num_classes=5, average="macro", thresholds=64).to(device)
    auprc       = MulticlassAveragePrecision(num_classes=5, average="macro", thresholds=64).to(device)
    conf_matrix = MulticlassConfusionMatrix(num_classes=5).to(device)

    with torch.no_grad():
        for batch in test_dl:
            ed_img, ed_mask, es_img, es_mask, info = batch
            y = info["group"].to(device)

            if model_name == "baseline":
                x = torch.cat([ed_img, es_img], dim=1).to(device)
                probs = torch.stack([m(x).softmax(dim=1) for m in models]).mean(dim=0)
            elif model_name == "from_mask":
                x = torch.cat([ed_mask, es_mask], dim=1).to(device)
                probs = torch.stack([m(x).softmax(dim=1) for m in models]).mean(dim=0)
            else:
                ed_mask_d = ed_mask.to(device)
                es_mask_d = es_mask.to(device)
                info_d = {k: v.to(device) if torch.is_tensor(v) else v for k, v in info.items()}
                probs = torch.stack([
                    m(m.normalize_features(m.get_features(ed_mask_d, es_mask_d, info_d))).softmax(dim=1)
                    for m in models
                ]).mean(dim=0)

            preds = probs.argmax(dim=1)
            acc_metric.update(preds, y)
            f1_metric.update(preds, y)
            f1_pc.update(preds, y)
            prec_pc.update(preds, y)
            rec_pc.update(preds, y)
            auroc.update(probs, y)
            auprc.update(probs, y)
            conf_matrix.update(preds, y)

    cm         = conf_matrix.compute().cpu().numpy()
    f1_vals    = f1_pc.compute()
    prec_vals  = prec_pc.compute()
    rec_vals   = rec_pc.compute()
    run_label  = f"{model_name}_{seg_variant}"

    run = wandb.init(
        project="acdc_classification",
        name=f"{run_label}_test_ensemble",
        group=f"{run_label}_test",
    )
    run.log({
        "test/acc":          acc_metric.compute().item(),
        "test/f1_macro":     f1_metric.compute().item(),
        "test/auroc_macro":  auroc.compute().item(),
        "test/auprc_macro":  auprc.compute().item(),
        **{f"test/f1_{CLASS_NAMES[i]}":   f1_vals[i].item()   for i in range(5)},
        **{f"test/prec_{CLASS_NAMES[i]}": prec_vals[i].item() for i in range(5)},
        **{f"test/rec_{CLASS_NAMES[i]}":  rec_vals[i].item()  for i in range(5)},
    })

    y_true, y_pred = [], []
    for ti in range(5):
        for pj in range(5):
            count = int(cm[ti, pj])
            y_true.extend([ti] * count)
            y_pred.extend([pj] * count)
    run.log({"test/confusion_matrix": wandb.plot.confusion_matrix(
        y_true=y_true, preds=y_pred, class_names=CLASS_NAMES)})

    results_dir.mkdir(parents=True, exist_ok=True)
    fig = _plot_conf_matrix(cm, CLASS_NAMES, save_path=results_dir / "conf_matrix_test_ensemble.pdf")
    run.log({"test/confusion_matrix_img": wandb.Image(fig)})
    plt.close(fig)
    _save_conf_matrices_json([cm], cm, CLASS_NAMES,
                             run, f"{run_label}_test_ensemble_conf_matrix")
    run.finish()

def master_run():
    extra_naming = "deterministic_final"
    results_base = Path("results/cross_val/classification")

    # baseline: cross_val only
    cross_val("baseline", lr=1e-3, extra_naming=extra_naming,
              save_info_path=results_base / "baseline" / "info.json")

    # from_mask: both
    cross_val("from_mask", lr=1e-3, extra_naming=extra_naming,
              save_info_path=results_base / "from_mask" / "info.json")
    cross_val("from_mask", lr=1e-3, extra_naming=extra_naming,
              save_info_path=results_base / "from_mask" / "info_seg.json", use_seg_masks=True)

    # medical_metrics_mlp: both
    cross_val("medical_metrics_mlp", lr=3e-4, extra_naming=extra_naming,
              save_info_path=results_base / "mlp" / "info.json")
    cross_val("medical_metrics_mlp", lr=3e-4, extra_naming=extra_naming,
              save_info_path=results_base / "mlp" / "info_seg.json", use_seg_masks=True)

    # medical_metrics_imn: both
    cross_val("medical_metrics_imn", lr=3e-4, extra_naming=extra_naming,
              save_info_path=results_base / "imn" / "info.json")
    cross_val("medical_metrics_imn", lr=3e-4, extra_naming=extra_naming,
              save_info_path=results_base / "imn" / "info_seg.json", use_seg_masks=True)

def test_master_run():
    # Step 1: generate ensemble segmentation masks for test patients
    test_ensemble_segmenter()

    # Step 2: classify test set with each model
    test_ensemble_classifier("baseline", seg_variant="gt")
    test_ensemble_classifier("from_mask", seg_variant="seg")
    test_ensemble_classifier("medical_metrics_mlp", seg_variant="seg")
    test_ensemble_classifier("medical_metrics_imn", seg_variant="seg")


if __name__ == "__main__":
    cross_segmenter2(save_info_path=Path(r"results\cross_val\segmentation\unet\info.json"))
    # master_run()
    # test_master_run()