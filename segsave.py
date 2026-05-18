import os
os.environ.setdefault("MIOPEN_FIND_MODE", "2")
os.environ["PYTORCH_MIOPEN_SUGGEST_NHWC"] = "0"
os.environ["MIOPEN_LOG_LEVEL"] = "2"

import torch
torch.set_float32_matmul_precision('high')

from lightning.pytorch import Trainer

from pathlib import Path
import numpy as np

from src.datamodules.acdc_dm_pred import ACDCPredDataModule
from src.acdc_segmentation2 import ACDCSegmentation2

from src.utils.logger import find_ckpts
from src.utils.transform import ACDC_Augmentations
from albumentations import CenterCrop

import cv2
cv2.setNumThreads(0)


def main():
    print("started prediction save")
    save_root = Path(r"data\ACDC\database\training_cross_val")
    save_root.mkdir(parents=True, exist_ok=True)
    
    ckpts = find_ckpts(Path(r"results\cross_val\segmentation\unet\info.json"))

    target_size = 256

    trainer = Trainer(
        precision="32-true",
        accelerator="gpu",
        devices=1,
        enable_model_summary=False,
    )


    for fold in range(5):
        ts = ACDC_Augmentations(CenterCrop(target_size,target_size,pad_if_needed=True),target_z=18)

        datamodule = ACDCPredDataModule(
            root=r"data\ACDC\database\training_npy",
            fold_idx=fold,
            val_transform=ts.get_replay_validation_augmentation(),
            batch_size=4,
        )
        model = ACDCSegmentation2.load_from_checkpoint(ckpts[fold],criterion=None)


        outputs = trainer.predict(model, datamodule=datamodule)

        for batch_out in outputs:
            patient_ids = batch_out["patient_id"]
            preds_ed = batch_out["preds_ed"]   # (B, Z, H, W)
            preds_es = batch_out["preds_es"]   # (B, Z, H, W)
            keep_ed = batch_out["keep_ed"]
            keep_es = batch_out["keep_es"]

            for i, pid in enumerate(patient_ids):
                patient_dir = save_root / str(pid)
                patient_dir.mkdir(parents=True, exist_ok=True)

                ed = preds_ed[i][keep_ed[i]].detach().cpu().numpy()
                es = preds_es[i][keep_es[i]].detach().cpu().numpy()

                np.save(patient_dir / f"ED_seg_{fold}.npy", ed)
                np.save(patient_dir / f"ES_seg_{fold}.npy", es)
                print(f"Saved fold {fold} for {pid}")
        print(f"prediction save ended for fold {fold}")

if __name__ == "__main__":
    main()