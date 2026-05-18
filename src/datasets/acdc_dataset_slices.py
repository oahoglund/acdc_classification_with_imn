import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json


class ACDC_SliceDataset(Dataset):
    """
    Slice-level dataset for ACDC.
    Memory-maps patient volumes and yields individual 2D slices.
    Returns (image, mask, info) per slice.
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
        dtype=np.float32,
    ):
        self.root = Path(root)
        if augmentation is None:
            raise ValueError("Augmentation cannot be None for this dataset")
        self.augmentation = augmentation
        self.dtype = dtype

        class_to_idx = {"DCM": 0, "HCM": 1, "MINF": 2, "NOR": 3, "RV": 4}

        self.samples = []
        for patient in patient_list:
            patient_dir = self.root / patient

            info_path = patient_dir / "info.json"
            with open(info_path, "r") as f:
                info_json = json.load(f)

            self.samples.append({
                "patient_id": patient,
                "ed_path": patient_dir / "ED_img.npy",
                "ed_gt_path": patient_dir / "ED_gt.npy",
                "es_path": patient_dir / "ES_img.npy",
                "es_gt_path": patient_dir / "ES_gt.npy",
                "info": {
                    "group": class_to_idx[info_json["group"]],
                    "height": float(info_json["height"]),
                    "weight": float(info_json["weight"]),
                    "spacing": info_json["image_meta"]["spacing"],
                }
            })

        if len(self.samples) == 0:
            raise RuntimeError("No samples found. Check patient list or root path.")

        # Cache mmap handles per patient
        self._mmap = []
        for s in self.samples:
            ed = np.load(s["ed_path"], mmap_mode="r")
            ed_gt = np.load(s["ed_gt_path"], mmap_mode="r")
            es = np.load(s["es_path"], mmap_mode="r")
            es_gt = np.load(s["es_gt_path"], mmap_mode="r")

            if ed.ndim != 3 or es.ndim != 3:
                raise RuntimeError(
                    f"Expected (Z,Y,X). Got ED {ed.shape}, ES {es.shape} for {s['patient_id']}"
                )
            if ed.shape[0] != es.shape[0]:
                raise RuntimeError(
                    f"ED/ES slice count mismatch for {s['patient_id']}: ED={ed.shape[0]}, ES={es.shape[0]}"
                )

            self._mmap.append({"ED": (ed, ed_gt), "ES": (es, es_gt)})

        # Build slice index: (patient_idx, phase, z)
        self.slice_index: list[tuple[int, str, int]] = []
        for p_idx in range(len(self.samples)):
            z_count = self._mmap[p_idx]["ED"][0].shape[0]
            for z in range(z_count):
                self.slice_index.append((p_idx, "ED", z))
                self.slice_index.append((p_idx, "ES", z))

    def __len__(self):
        return len(self.slice_index)

    @staticmethod
    def _to_hwc_1ch(slice_yx: np.ndarray) -> np.ndarray:
        return slice_yx[..., None]  # (H,W) -> (H,W,1)

    def __getitem__(self, i: int):
        p_idx, phase, z = self.slice_index[i]
        sample = self.samples[p_idx]

        vol, gt_vol = self._mmap[p_idx][phase]
        img = vol[z].astype(self.dtype, copy=False)    # (H,W)
        msk = gt_vol[z].astype(np.int16, copy=False)   # (H,W)

        augmented = self.augmentation(
            image=self._to_hwc_1ch(img),
            mask=self._to_hwc_1ch(msk),
        )

        x = augmented["image"].float()  # (1,H,W)
        y = augmented["mask"]
        if y.ndim == 3 and y.shape[0] == 1:
            y = y[0]
        y = y.long()  # (H,W)

        info = {
            "patient_id": sample["patient_id"],
            "phase": phase,
            "slice_idx": torch.tensor(z, dtype=torch.long),
            "group": torch.tensor(sample["info"]["group"], dtype=torch.long),
            "height": torch.tensor(sample["info"]["height"], dtype=torch.float32),
            "weight": torch.tensor(sample["info"]["weight"], dtype=torch.float32),
            "spacing": torch.tensor(sample["info"]["spacing"], dtype=torch.float32),
        }

        return x, y, info
