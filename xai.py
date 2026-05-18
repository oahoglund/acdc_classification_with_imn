import os
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "0"
os.environ["MIOPEN_LOG_LEVEL"] = "2"

SEED = 67

import torch
torch.set_float32_matmul_precision('highest')

from lightning.pytorch import seed_everything, Trainer

seed_everything(SEED, workers=True)
torch.use_deterministic_algorithms(True)

from pathlib import Path
import numpy as np

from src.datamodules.acdc_dm import ACDCDataModule
from src.clas_medical_metrics_imn import ClassificationMedicalMetricsIMN
from src.datasets.acdc_dataset_patients import ACDC_PatientDataset
from src.utils.transform import ACDC_Augmentations
from src.utils.logger import find_ckpts
from albumentations import CenterCrop
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import seaborn as sns
import cv2
cv2.setNumThreads(0)

CLASS_NAMES    = ["DCM", "HCM", "MINF", "NOR", "ARV"]
N_CLASSES      = len(CLASS_NAMES)
FEATURE_LABELS = ["LVEF", "RVEF", "LVEDV", "RVEDV", "MYmass"]

CKPTS_SEG = find_ckpts(Path(r"results\cross_val\classification\imn\info_seg.json"))

mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (4.5, 3.2),
    "text.usetex": False,
})
sns.set_theme(style="whitegrid", context="paper")


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_fold_contributions(outputs):
    """
    Returns contributions for all samples (not filtered by correctness).

    Returns
    -------
    contrib_by_class : dict[int -> Tensor]
        For each TRUE class index, a (N_c, N_features) tensor of signed
        contributions  features * W[pred_class].
    global_contrib : Tensor (N_samples, N_features)
        Signed contributions for ALL samples.
    targets, preds : 1-D LongTensors
    """
    features = torch.cat([o["features"] for o in outputs], dim=0)   # (N, F)
    preds    = torch.cat([o["preds"]    for o in outputs], dim=0)   # (N,)
    targets  = torch.cat([o["targets"]  for o in outputs], dim=0)   # (N,)
    W        = torch.cat([o["W"]        for o in outputs], dim=0)   # (N, C, F)

    pred_W  = W[torch.arange(W.size(0)), preds]   # (N, F)
    contrib = features * pred_W                    # (N, F) signed

    contrib_by_class = {}
    for c in range(N_CLASSES):
        mask = (targets == c) & (preds == c)   # correctly classified only
        if mask.any():
            contrib_by_class[c] = contrib[mask]

    return contrib_by_class, contrib, targets, preds


def aggregate_importance(fold_results):
    """
    Pools contributions from all folds/models and computes mean |contribution|.

    Returns
    -------
    per_class_importance : dict[int -> np.ndarray]  shape (F,)
    global_importance    : np.ndarray               shape (F,)
    """
    per_class_all = {c: [] for c in range(N_CLASSES)}
    global_all    = []

    for fold_dict, global_contrib in fold_results:
        global_all.append(global_contrib.numpy())
        for c, tensor in fold_dict.items():
            per_class_all[c].append(tensor.numpy())

    per_class_importance = {}
    for c in range(N_CLASSES):
        if per_class_all[c]:
            stacked = np.concatenate(per_class_all[c], axis=0)
            per_class_importance[c] = stacked.mean(axis=0)

    global_all_stacked   = np.concatenate(global_all, axis=0)
    global_importance     = global_all_stacked.mean(axis=0)          # signed mean
    global_importance_abs = np.abs(global_all_stacked).mean(axis=0)  # mean of |x|
    return per_class_importance, global_importance, global_importance_abs


# ── plotting ──────────────────────────────────────────────────────────────────

BAR_COLOR = "#4C72B0"


def _bar_chart(importances, save_path, feature_labels, ylim=None):
    fig, ax = plt.subplots()
    x = np.arange(len(importances))
    ax.bar(x, importances, color=BAR_COLOR, edgecolor="black", linewidth=0.6)
    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_labels, rotation=0)
    ax.set_ylabel("Mean contribution")
    ax.grid(False)
    ax.yaxis.grid(True, alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ylim is not None:
        ax.set_ylim(ylim)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


def save_plots(per_class_importance, global_importance, global_importance_abs, feature_labels, save_root):
    plots_dir = save_root / "xai_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _bar_chart(global_importance,     plots_dir / "importance_global_signed.pdf", feature_labels)
    _bar_chart(global_importance_abs, plots_dir / "importance_global_abs.pdf",    feature_labels,
               ylim=(0, 2))

    for c, name in enumerate(CLASS_NAMES):
        if c not in per_class_importance:
            print(f"  [skip] No samples for class {name}")
            continue
        _bar_chart(per_class_importance[c],
                   plots_dir / f"importance_{name}.pdf", feature_labels,
                   ylim=(-1.3, 5.3))


def save_csv(per_class_importance, global_importance, global_importance_abs, feature_labels, save_root):
    rows = {"feature": feature_labels, "global_signed": global_importance, "global_abs": global_importance_abs}
    for c, name in enumerate(CLASS_NAMES):
        rows[name] = per_class_importance.get(c, np.full(len(feature_labels), np.nan))
    df = pd.DataFrame(rows).set_index("feature")
    csv_path = save_root / "xai_feature_importance.csv"
    df.to_csv(csv_path, float_format="%.6f")
    print(f"  Saved → {csv_path}")


# ── cross-val XAI (validation patients, seg-mask variant) ─────────────────────

def main():
    print("Starting XAI analysis for IMN — cross-val (seg masks)")
    save_root = Path(r"results\cross_val\classification\imn")
    save_root.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(precision="32-true", accelerator="gpu", devices=1,
                      enable_model_summary=False)

    in_channels = 18 * 2
    target_size = 256
    criterion   = torch.nn.CrossEntropyLoss(weight=torch.ones(N_CLASSES))

    fold_results = []

    for fold_idx in range(5):
        print(f"\n── Fold {fold_idx} ──────────────────────────────")
        seed_everything(SEED + fold_idx, workers=True)

        ts = ACDC_Augmentations(
            CenterCrop(target_size, target_size, pad_if_needed=True),
            target_z=in_channels // 2,
        )
        datamodule = ACDCDataModule(
            root=r"data\ACDC\database\training_npy",
            fold_idx=fold_idx,
            train_transform=ts.get_training_augmentation(),
            val_transform=ts.get_validation_augmentation(),
            batch_size=6,
            slice_mode=False,
            use_fold_masks=True,
        )

        datamodule.setup("fit")
        model = ClassificationMedicalMetricsIMN.load_from_checkpoint(
            CKPTS_SEG[fold_idx], criterion=criterion
        )

        outputs = trainer.predict(model, dataloaders=datamodule.val_dataloader())
        contrib_by_class, global_contrib, targets, preds = compute_fold_contributions(outputs)

        acc = (preds == targets).float().mean().item()
        print(f"  Fold accuracy: {acc:.3f}")

        fold_results.append((contrib_by_class, global_contrib))

    print("\nAggregating across folds…")
    per_class_importance, global_importance, global_importance_abs = aggregate_importance(fold_results)

    print("\nSaving plots…")
    save_plots(per_class_importance, global_importance, global_importance_abs, FEATURE_LABELS, save_root)

    print("\nSaving CSV…")
    save_csv(per_class_importance, global_importance, global_importance_abs, FEATURE_LABELS, save_root)

    print("\nDone! Results in:", save_root)


# ── test set XAI (50 held-out patients, ensemble masks) ───────────────────────

def main_test():
    print("Starting XAI analysis for IMN — test set (ensemble masks)")
    save_root = Path(r"results\test_set\classification\medical_metrics_imn_seg")
    save_root.mkdir(parents=True, exist_ok=True)

    trainer    = Trainer(precision="32-true", accelerator="gpu", devices=1,
                         enable_model_summary=False)
    criterion  = torch.nn.CrossEntropyLoss(weight=torch.ones(N_CLASSES))

    target_size  = 256
    test_npy_root = Path(r"data\ACDC\database\testing_npy")
    test_seg_root = Path(r"data\ACDC\database\testing_cross_val")

    test_patients = sorted([p.name for p in test_npy_root.iterdir() if p.is_dir()])

    ts = ACDC_Augmentations(
        CenterCrop(target_size, target_size, pad_if_needed=True),
        target_z=18,
    )
    test_ds = ACDC_PatientDataset(
        root=test_npy_root,
        patient_list=test_patients,
        augmentation=ts.get_validation_augmentation(),
        use_fold_masks=True,
        seg_root_override=test_seg_root,
        seg_suffix="ensemble",
    )
    test_dl = DataLoader(test_ds, batch_size=6, shuffle=False, num_workers=4,
                         pin_memory=True, persistent_workers=True, prefetch_factor=2)

    # run each of the 5 models over all 50 test patients and aggregate contributions
    fold_results = []
    for ckpt_idx, ckpt in enumerate(CKPTS_SEG):
        print(f"\n── Model {ckpt_idx} ({Path(ckpt).parent.name}) ──────────")
        model   = ClassificationMedicalMetricsIMN.load_from_checkpoint(ckpt, criterion=criterion)
        outputs = trainer.predict(model, dataloaders=test_dl)

        contrib_by_class, global_contrib, targets, preds = compute_fold_contributions(outputs)

        acc = (preds == targets).float().mean().item()
        print(f"  Accuracy: {acc:.3f}")

        fold_results.append((contrib_by_class, global_contrib))

    print("\nAggregating across models…")
    per_class_importance, global_importance, global_importance_abs = aggregate_importance(fold_results)

    print("\nSaving plots…")
    save_plots(per_class_importance, global_importance, global_importance_abs, FEATURE_LABELS, save_root)

    print("\nSaving CSV…")
    save_csv(per_class_importance, global_importance, global_importance_abs, FEATURE_LABELS, save_root)

    print("\nDone! Results in:", save_root)


if __name__ == "__main__":
    main()        # cross-val XAI
    main_test()   # test set XAI
