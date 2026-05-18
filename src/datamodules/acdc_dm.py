from typing import Optional

import lightning as L
from torch.utils.data import DataLoader
from pathlib import Path
import json

from src.datasets.acdc_dataset_slices import ACDC_SliceDataset
from src.datasets.acdc_dataset_patients import ACDC_PatientDataset


class ACDCDataModule(L.LightningDataModule):
    """
    Unified ACDC data module.

    Parameters
    ----------
    root : str
        Path to the training_npy directory.
    slice_mode : bool
        True  → use ACDC_SliceDataset (one 2D slice per item, for segmentation training).
        False → use ACDC_PatientDataset (one 3D patient per item, for classification).
    use_fold_masks : bool
        Only used when slice_mode=False.
        False → load ground-truth masks (ED_gt.npy / ES_gt.npy).
        True  → load cross-val predicted masks (ED_seg_<fold>.npy / ES_seg_<fold>.npy).
    seg_root_override : str, optional
        Override the directory from which fold masks are loaded.
        Defaults to <root>/../training_cross_val.
    seg_suffix : str, optional
        Override the fold suffix in mask file names (e.g. "ensemble").
    """

    def __init__(
        self,
        root: str,
        train_transform,
        val_transform,
        fold_idx: Optional[int] = None,
        slice_mode: bool = True,
        use_fold_masks: bool = False,
        seg_root_override: str = None,
        seg_suffix: str = None,
        batch_size: int = 16,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_mem: bool = True,
        persistent_workers: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        if fold_idx is None:
            raise ValueError("fold_idx must be provided")
        self.fold_idx = fold_idx
        self.slice_mode = slice_mode
        self.use_fold_masks = use_fold_masks
        self.seg_root_override = seg_root_override
        self.seg_suffix = seg_suffix
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_mem = pin_mem
        self.persistent_workers = persistent_workers
        self.drop_last = drop_last

    def _load_split(self) -> tuple[list[str], list[str]]:
        with open(self.root / "splits.json") as f:
            split = json.load(f)
        fold = split[f"fold_{self.fold_idx}"]
        return fold["train"], fold["val"]

    def _make_dataset(self, patient_list, transform):
        if self.slice_mode:
            return ACDC_SliceDataset(self.root, patient_list=patient_list, augmentation=transform)
        else:
            return ACDC_PatientDataset(
                self.root,
                patient_list=patient_list,
                augmentation=transform,
                fold=self.fold_idx,
                use_fold_masks=self.use_fold_masks,
                seg_root_override=self.seg_root_override,
                seg_suffix=self.seg_suffix,
            )

    def setup(self, stage: str):
        train_cases, val_cases = self._load_split()
        print(f"Loaded fold {self.fold_idx} with {len(train_cases)} train cases and {len(val_cases)} val cases")

        if stage == "fit":
            self.train_ds = self._make_dataset(train_cases, self.train_transform)
            self.val_ds   = self._make_dataset(val_cases,   self.val_transform)

        if stage == "validate":
            self.val_ds = self._make_dataset(val_cases, self.val_transform)

        if stage == "test":
            self.test_ds = self._make_dataset(val_cases, self.val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor, pin_memory=self.pin_mem,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor, pin_memory=self.pin_mem,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor, pin_memory=self.pin_mem,
        )
