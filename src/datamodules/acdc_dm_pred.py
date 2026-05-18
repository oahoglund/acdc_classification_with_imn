from typing import Optional

import lightning as L
from torch.utils.data import DataLoader
from pathlib import Path
import json

from src.datasets.acdc_dataset_pred import ACDC_PredDataset


class ACDCPredDataModule(L.LightningDataModule):
    """
    Data module for running segmentation prediction (segsave).
    Uses ACDC_PredDataset — no ground-truth masks required.
    """

    def __init__(
        self,
        root: str,
        val_transform,
        fold_idx: Optional[int] = None,
        batch_size: int = 16,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_mem: bool = True,
        persistent_workers: bool = True,
    ):
        super().__init__()
        self.root = Path(root)
        if fold_idx is None:
            raise ValueError("fold_idx must be provided")
        self.fold_idx = fold_idx
        self.val_transform = val_transform
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.pin_mem = pin_mem
        self.persistent_workers = persistent_workers

    def _load_split(self) -> tuple[list[str], list[str]]:
        with open(self.root / "splits.json") as f:
            split = json.load(f)
        fold = split[f"fold_{self.fold_idx}"]
        return fold["train"], fold["val"]

    def setup(self, stage: str):
        train_cases, val_cases = self._load_split()
        print(f"Loaded fold {self.fold_idx} with {len(train_cases)} train cases and {len(val_cases)} val cases")

        if stage == "predict":
            full_list = val_cases + train_cases
            self.pred_ds = ACDC_PredDataset(self.root, patient_list=full_list, augmentation=self.val_transform)

    def predict_dataloader(self):
        return DataLoader(
            self.pred_ds, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor, pin_memory=self.pin_mem,
        )
