import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Union
import json


class ACDC_Dataset_Seg(Dataset):
    CLASSES = [
        "background",
        "right ventricle",
        "myocardium",
        "left ventricle"
    ]

    def __init__(
        self,
        root: Path,
        patient_list: list[str],
        augmentation,
        dtype=np.float32,
    ):
        # patient list contains patient IDs like "patient001", "patient002", etc.
        self.root = Path(root) # training_npy for instance
        if augmentation is None:
            raise ValueError("Augmentation cannot be None for this dataset")
        self.augmentation = augmentation
        self.dtype = dtype

        self.samples = []
        for patient in patient_list:
            patient_dir = self.root / patient

            # I want to save the img and msk paths, but also config info (group, height, weight)
            ed_path = patient_dir / f"ED_img.npy"
            ed_gt_path = patient_dir / f"ED_gt.npy"
            es_path = patient_dir / f"ES_img.npy"
            es_gt_path = patient_dir / f"ES_gt.npy"
            info_path = patient_dir / "info.json"

            self.samples.append({
                "patient_id": patient,
                "ed_path": ed_path,
                "ed_gt_path": ed_gt_path,
                "es_path": es_path,
                "es_gt_path": es_gt_path,
                "info_path": info_path
            })

        if len(self.samples) == 0:
            raise RuntimeError("No .npy samples found. Check out patient list or path")

    def __len__(self):
        return len(self.samples)
    
    def _fix_meta_from_replay(self, replay:dict):
        meta = None
        for t in replay["transforms"]:
            name = t.get("__class_fullname__", "")
            if name in ("src.utils.transform.GroundTruthMarginCrop", "src.utils.transform.GroundTruthCrop"):
                raise NotImplementedError("You shouldnt use ground truth crop yet, as this is not safely implemented for ACDC")
            elif name.endswith("CenterCrop") or name == "CenterCrop":
                meta = self._meta_from_centercrop_transform(t)
        
        if meta is None:
            names = [t.get("__class_fullname__", "") for t in replay["transforms"]]
            raise ValueError(f"No invertible spatial transform found in replay to build meta. {names}")
        return meta
    
    @staticmethod
    def _meta_from_groundtruthcrop_transform(t: dict) -> dict:
        params = t.get("params", {})
        H, W, _C = params["shape"]  # original HWC at that time
        # Enforce required keys (adjust if your transform uses different names)
        return {
            "orig_h": int(H),
            "orig_w": int(W),
            "x1": int(params["x1"]),
            "y1": int(params["y1"]),
            "x2": int(params["x2"]),
            "y2": int(params["y2"]),
            "pad_left": int(params.get("pad_left", 0)),
            "pad_top": int(params.get("pad_top", 0)),
            "pad_right": int(params.get("pad_right", 0)),
            "pad_bottom": int(params.get("pad_bottom", 0)),
        }
    
    @staticmethod
    def _meta_from_centercrop_transform(t: dict) -> dict:
        params = t.get("params", {})
        H, W, _C = params["shape"]  # original HWC at that time
        x1, y1, x2, y2 = params["crop_coords"]

        pad_params = params.get("pad_params") or {}
        # Albumentations may store these keys; if no pad happened, pad_params is None
        pad_left = int(pad_params.get("pad_left", 0))
        pad_top = int(pad_params.get("pad_top", 0))
        pad_right = int(pad_params.get("pad_right", 0))
        pad_bottom = int(pad_params.get("pad_bottom", 0))

        # Fix x1, y1, x2, y2 to refer to original coordinates rather than the cropped ones (without the padding)
        x2 = x2 - pad_left - pad_right
        y2 = y2 - pad_top - pad_bottom

        return {
            "orig_h": int(H),
            "orig_w": int(W),
            "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
            "pad_left": pad_left, "pad_top": pad_top, "pad_right": pad_right, "pad_bottom": pad_bottom,
        }

    def __getitem__(self, i):
        def to_hwc(vol):
            return np.transpose(vol, (1, 2, 0))  # (Y, X, Z)

        
        sample = self.samples[i]
        ed_img = np.load(sample["ed_path"]).astype(self.dtype)    # (Z,Y,X)
        ed_gt = np.load(sample["ed_gt_path"]).astype(np.int16)      # (Z,Y,X)
        es_img = np.load(sample["es_path"]).astype(self.dtype)    #
        es_gt = np.load(sample["es_gt_path"]).astype(np.int16)      #


        ed_img_hwc = to_hwc(ed_img)
        ed_gt_hwc  = to_hwc(ed_gt)
        es_img_hwc = to_hwc(es_img)
        es_gt_hwc  = to_hwc(es_gt)

        # assumes that augmentation has toTensorV2
        #print(f"Started augmentation: {i}")
        augmented = self.augmentation(
            image=ed_img_hwc,
            mask=ed_gt_hwc,
            image2=es_img_hwc,
            mask2=es_gt_hwc,
        )
        #print(type(augmented["image"]), augmented["image"].shape)
        ed_img_aug = augmented["image"].float()
        ed_gt_aug  = augmented["mask"].long()
        es_img_aug = augmented["image2"].float()
        es_gt_aug  = augmented["mask2"].long()
        meta = self._fix_meta_from_replay(augmented["replay"])

        # meta["img_fp"] = str(img_fp)
        with open(sample["info_path"], "r") as f:
            info = json.load(f)

        class_to_idx = {
            "DCM": 0,
            "HCM": 1,
            "MINF": 2,
            "NOR": 3,
            "RV": 4
        }
        label = torch.tensor(class_to_idx[info["group"]], dtype=torch.long)
        # patient_id": sample["patient_id"],
        info = {
            "patient_id": str(sample["patient_id"]),
            "group": torch.tensor(class_to_idx[info["group"]], dtype=torch.long),
            "height": torch.tensor(info["height"], dtype=torch.float32),
            "weight": torch.tensor(info["weight"], dtype=torch.float32),
            "spacing": torch.tensor(
                info["image_meta"]["spacing"], dtype=torch.float32
            ),

        }
        return ed_img_aug, es_img_aug, info, meta