from tqdm import tqdm
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import json
from sklearn.model_selection import StratifiedKFold
from typing import Union
from collections import Counter

"""
This script performs the following steps:
1. Lists valid cases in the ACDC dataset (cases that have both image and mask files).
2. Splits the cases into training, validation using K-Fold cross-validation. (Stratified by the "Group" label to maintain class balance across folds).
3. Preprocesses the images and masks by resampling them to a specified spacing and saves them as .npy files. Also saves relevant metadata (spacing, origin, direction) and clinical info (group, height, weight) in a JSON file for each case.
4. Saves the cross-validation splits and class frequencies in a JSON file for later use.
"""

"""
Saved files pr patient:
(ED = End Diastole, ES = End Systole, img = MRI of heart, gt = Ground truth mask)
(They are saved as numpy arrays with shape (z, y, x) and spacing 1.6mm in x and y)
- ED_img.npy
- ED_gt.npy
- ES_img.npy
- ES_gt.npy

- info.json
    {
        "group": str # one of "NOR", "DCM", "HCM", "MINF", "RV"
        "height": float,
        "weight": float,
    }
    {
        "spacing": (dx, dy, dz),
        "origin": (ox, oy, oz),
        "direction": (d00, d01, d02, d10, d11, d12, d20, d21, d22)
    }
"""

"""
splits.json
{
    "fold_x": {
        "train": [list of patient IDs],
        "val": [list of patient IDs],
        "train_class_freq": {"NOR": int, "DCM": int, "HCM": int, "MINF": int, "RV": int},
        "val_class_freq": {"NOR": int, "DCM": int, "HCM": int, "MINF": int, "RV": int},
    },
"""


def read_config(dir_filepath: Union[str, Path]):
        schema = {
            "ED": int,
            "ES": int,
            "Group": str,
            "Height": float,
            "NbFrame": int,
            "Weight": float,
        }

        config = {}
        dir_filepath = Path(dir_filepath)

        with open(dir_filepath / "Info.cfg") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                key, value = line.split(":", 1)
                key = key.strip()
                value = value.strip()

                cast = schema.get(key, str)
                config[key] = cast(value)

        return config

def get_patient_list_and_y(root: Path):
    patient_list = []
    y = []
    for i in range(1,101):
        patient_name = f"patient{i:03d}"
        path_to_patient = root / patient_name
        conf = read_config(path_to_patient)
        patient_list.append(patient_name)
        y.append(conf["Group"])
    return patient_list, y

def cross_validation_split(patient_list: list[list], y: list[str], folds:int = 5, seed:int = 42):
    cases = patient_list.copy()

    kf = StratifiedKFold(n_splits=folds, random_state=seed, shuffle = True)
    folds_data = {}

    for i, (train_idx, val_idx) in enumerate(kf.split(cases, y)):
        train_cases = [cases[i] for i in train_idx]
        val_cases = [cases[i] for i in val_idx]

        train_labels = [y[j] for j in train_idx]
        val_labels = [y[j] for j in val_idx]

        folds_data[f"fold_{i}"] = {
            "train": train_cases,
            "val": val_cases,
            "train_class_freq": dict(Counter(train_labels)),
            "val_class_freq": dict(Counter(val_labels)),
        }

    return folds_data


def save_cross_split(folds_data:dict, filepath: Path):
    with open(filepath, "w") as f:
        json.dump(folds_data, f, indent=2)


def preprocess_to_npy(patient_list: list[str], root: Path, out_dir: Path, spacing:float = 1.0):
    out_dir.mkdir(exist_ok=True)

    for patient in tqdm(patient_list):
        patient_dir = root / patient
        conf = read_config(patient_dir)

        out_patient_dir = out_dir / patient
        out_patient_dir.mkdir(exist_ok=True)

        for phase in ["ED", "ES"]:
            frame = f"frame{conf[phase]:02d}"
            img_path = patient_dir / f"{patient}_{frame}.nii.gz"
            gt_path = patient_dir / f"{patient}_{frame}_gt.nii.gz"

            img_stk = resample_xy_to_mm(img_path, is_label=False,spacing=spacing)
            img = sitk_to_numpy(img_stk).astype(np.float32)

            gt_stk = resample_xy_to_mm(gt_path, is_label=True,spacing=spacing) # (x,y,z)
            gt = sitk_to_numpy(gt_stk).astype(np.int16) # (z, y, x)
            np.save(out_patient_dir / f"{phase}_img.npy", img)
            np.save(out_patient_dir / f"{phase}_gt.npy", gt)
        
        image_meta = {
            "spacing": img_stk.GetSpacing(),
            "origin": img_stk.GetOrigin(),
            "direction": img_stk.GetDirection(),
        }
        out_dict = {
            "group": conf["Group"],
            "height": conf["Height"],
            "weight": conf["Weight"],
            "image_meta": image_meta,
        }

        with open(out_patient_dir / "info.json", "w") as f:
            json.dump(out_dict, f, indent=2)

import SimpleITK as sitk


def resample_xy_to_mm(nifti_path, is_label=False, spacing=1.0, default_value=0):
    img = sitk.ReadImage(nifti_path)

    original_spacing = img.GetSpacing()  # (sx, sy, sz)
    original_size    = img.GetSize()     # (x, y, z)

    new_spacing = (float(spacing), float(spacing), original_spacing[2])

    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
        original_size[2],
    ]
    new_size = [max(1, s) for s in new_size]

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img)         
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_value)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        img = sitk.Cast(img, sitk.sitkUInt8)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(img)

def sitk_to_numpy(img):
    return sitk.GetArrayFromImage(img)  # (z, y, x)


def main():
    seed = 67
    root = Path(r"data\ACDC\database\training")
    out = Path(r"data\ACDC\database\training_npy")
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    # Split dataset into train and validation and test sets
    patient_list, y = get_patient_list_and_y(root)
    folds_data = cross_validation_split(patient_list, y, folds=5, seed=seed)

    preprocess_to_npy(patient_list, root, out_dir=out, spacing=1.6)

    save_cross_split(folds_data, filepath=out/"splits.json")

def preprocess_test():
    seed = 67
    root = Path(r"data\ACDC\database\testing")
    out = Path(r"data\ACDC\database\testing_npy")
    patient_list = [f"patient{i:03d}" for i in range(101,151)]
    preprocess_to_npy(patient_list, root, out_dir=out, spacing=1.6)


if __name__ == "__main__":
    main()
    preprocess_test()