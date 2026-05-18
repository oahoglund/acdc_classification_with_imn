"""
Generate confusion matrix PDFs from saved JSON files.
Edit the MATRICES list below to control which ones are plotted and what titles they get.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ── configure which matrices to plot ──────────────────────────────────────────
# Each entry: (json_path, title, output_pdf)
# Use "aggregated" key from the JSON (sum across folds).

results = Path("results/test_set")

MATRICES = [
    (
        results / "segmentation/unet/unet_resnet34_dice_lr1e-4_cosineanealing_dataloader2_patience40_test_ensemble_conf_matrix.json",
        results / "segmentation/unet/conf_unet.pdf",
    ),
    (
        results / "classification/baseline_gt/baseline_gt_test_ensemble_conf_matrix.json",
        results / "classification/baseline_gt/conf_baseline.pdf",
    ),
    (
        results / "classification/from_mask_seg/from_mask_seg_test_ensemble_conf_matrix.json",
        results / "classification/from_mask_seg/conf_cnn_masks.pdf",
    ),
    (
        results / "classification/medical_metrics_mlp_seg/medical_metrics_mlp_seg_test_ensemble_conf_matrix.json",
        results / "classification/medical_metrics_mlp_seg/conf_mlp.pdf",
    ),
    (
        results / "classification/medical_metrics_imn_seg/medical_metrics_imn_seg_test_ensemble_conf_matrix.json",
        results / "classification/medical_metrics_imn_seg/conf_imn.pdf",
    ),
]


# ── plotting ───────────────────────────────────────────────────────────────────

def plot_conf_matrix(cm: np.ndarray, class_names: list, save_path: Path = None):
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm / row_sums, 0.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1.0)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    thresh = 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            prop  = cm_norm[i, j]
            color = "white" if prop > thresh else "black"
            ax.text(j, i, f"{prop:.2f}",
                    ha="center", va="center",
                    fontsize=12, color=color)

    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved → {save_path}")
    return fig


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    for json_path, out_pdf in MATRICES:
        json_path = Path(json_path)
        if not json_path.exists():
            print(f"[skip] Not found: {json_path}")
            continue

        with open(json_path) as f:
            data = json.load(f)

        cm          = np.array(data["aggregated"])
        class_names = data["class_names"]

        fig = plot_conf_matrix(cm, class_names, Path(out_pdf))
        plt.close(fig)

    print("Done.")
