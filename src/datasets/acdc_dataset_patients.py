import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json


class ACDC_PatientDataset(Dataset):
    """
    Patient-level dataset for ACDC.
    Returns full 3D volumes (ED + ES) per patient.
    Returns (ed_img, ed_mask, es_img, es_mask, info).

    use_fold_masks=False (default): loads ground-truth masks from ED_gt.npy / ES_gt.npy
    use_fold_masks=True:            loads cross-val predicted masks from
                                    <seg_root>/<patient>/ED_seg_<seg_name>.npy
                                    where seg_name = seg_suffix or str(fold)
    """
    CLASSES = [
        "background",
        "right ventricle",
        "myocardium",
        "left ventricle",
    ]

    def __init__(
        self,
        root: Path,
        patient_list: list[str],
        augmentation,
        fold: int = None,
        use_fold_masks: bool = False,
        seg_root_override: Path = None,
        seg_suffix: str = None,
        dtype=np.float32,
    ):
        # root is always the training_npy directory
        self.img_root = Path(root)
        if augmentation is None:
            raise ValueError("Augmentation cannot be None for this dataset")
        self.augmentation = augmentation
        self.dtype = dtype

        if use_fold_masks:
            if seg_root_override is not None:
                self.seg_root = Path(seg_root_override)
            else:
                # default: sibling directory training_cross_val
                self.seg_root = self.img_root.parent / "training_cross_val"
            seg_name = seg_suffix if seg_suffix is not None else str(fold)
        else:
            seg_name = None  # unused

        self.use_fold_masks = use_fold_masks
        self.seg_name = seg_name

        self.samples = []
        for patient in patient_list:
            img_dir = self.img_root / patient

            self.samples.append({
                "patient_id": patient,
                "ed_path": img_dir / "ED_img.npy",
                "es_path": img_dir / "ES_img.npy",
                "ed_mask_path": (
                    (self.seg_root / patient / f"ED_seg_{seg_name}.npy")
                    if use_fold_masks
                    else (img_dir / "ED_gt.npy")
                ),
                "es_mask_path": (
                    (self.seg_root / patient / f"ES_seg_{seg_name}.npy")
                    if use_fold_masks
                    else (img_dir / "ES_gt.npy")
                ),
                "info_path": img_dir / "info.json",
            })

        if len(self.samples) == 0:
            raise RuntimeError("No samples found. Check patient list or root path.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        def to_hwc(vol):
            return np.transpose(vol, (1, 2, 0))  # (Z,Y,X) -> (Y,X,Z)

        sample = self.samples[i]
        ed_img  = np.load(sample["ed_path"]).astype(self.dtype)
        ed_mask = np.load(sample["ed_mask_path"]).astype(np.int16)
        es_img  = np.load(sample["es_path"]).astype(self.dtype)
        es_mask = np.load(sample["es_mask_path"]).astype(np.int16)

        augmented = self.augmentation(
            image=to_hwc(ed_img),
            mask=to_hwc(ed_mask),
            image2=to_hwc(es_img),
            mask2=to_hwc(es_mask),
        )

        ed_img_aug  = augmented["image"].float()
        ed_mask_aug = augmented["mask"].long()
        es_img_aug  = augmented["image2"].float()
        es_mask_aug = augmented["mask2"].long()

        with open(sample["info_path"], "r") as f:
            info_json = json.load(f)

        class_to_idx = {"DCM": 0, "HCM": 1, "MINF": 2, "NOR": 3, "RV": 4}
        info = {
            "patient_id": str(sample["patient_id"]),
            "group": torch.tensor(class_to_idx[info_json["group"]], dtype=torch.long),
            "height": torch.tensor(info_json["height"], dtype=torch.float32),
            "weight": torch.tensor(info_json["weight"], dtype=torch.float32),
            "spacing": torch.tensor(info_json["image_meta"]["spacing"], dtype=torch.float32),
        }

        return ed_img_aug, ed_mask_aug, es_img_aug, es_mask_aug, info
