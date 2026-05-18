import cv2
import numpy as np
import albumentations as A
import torch

class ForegroundPercentileClip(A.ImageOnlyTransform):
    """
    Clip image values to foreground percentile range.
    """
    def __init__(self, background_value=0, low=1, high=99, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.bg = background_value
        self.low = low
        self.high = high

    def apply(self, img, **params):
        img = img.astype(np.float32)

        for c in range(img.shape[2]):
            channel = img[..., c]
            mask = channel != self.bg
            if not np.any(mask):
                continue
            p_low, p_high = np.percentile(channel[mask], [self.low, self.high])
            clipped = channel.copy()
            clipped[mask] = np.clip(channel[mask], p_low, p_high)
            img[..., c] = clipped

        return img


class MaskedNormalize(A.ImageOnlyTransform):
    """
    Normalize image by computing mean and std over foreground pixels only.
    """
    def __init__(self, background_value=0, eps=1e-6, always_apply=True, p=1.0):
        super().__init__(always_apply, p)
        self.bg = background_value
        self.eps = eps

    def apply(self, img, **params):
        img = img.astype(np.float32)

        # Expect HWC
        if img.ndim != 3:
            return img

        for c in range(img.shape[2]):
            channel = img[..., c]
            mask = channel != self.bg

            if not np.any(mask):
                continue

            fg = channel[mask]
            mean = fg.mean()
            std = fg.std()

            if std < self.eps:
                channel[mask] = 0.0
            else:
                channel[mask] = (channel[mask] - mean) / std

            # keep padding/background clean and consistent
            channel[~mask] = 0.0
            img[..., c] = channel

        return img


class PadZTo(A.DualTransform):
    def __init__(self, target_z=10, value=0, mask_value=0, p=1.0):
        super().__init__(p=p)
        self.target_z = target_z
        self.value = value
        self.mask_value = mask_value

    def get_params_dependent_on_data(self, params, data):
        # data contains all inputs passed to Compose: image, mask, image2, mask2, etc. :contentReference[oaicite:1]{index=1}
        img = data["image"]  # HWC
        z = img.shape[2]

        # If you are using additional_targets={"image2":"image"}, this transform will be applied to image2 too.
        # Sharing params only makes sense if Z matches.
        if "image2" in data:
            z2 = data["image2"].shape[2]
            if z2 != z:
                raise ValueError(
                    f"PadZTo (shared params) requires image and image2 to have same Z. "
                    f"Got Z(image)={z}, Z(image2)={z2}. Use per-input padding (Option A) instead."
                )

        if z == self.target_z:
            return {"mode": "none", "z0": 0, "z1": 0, "pad_left": 0, "pad_right": 0}

        if z > self.target_z:
            start = (z - self.target_z) // 2
            return {"mode": "crop", "z0": start, "z1": start + self.target_z, "pad_left": 0, "pad_right": 0}

        # z < target_z -> pad
        pad = self.target_z - z
        pad_left = pad // 2
        pad_right = pad - pad_left
        return {"mode": "pad", "z0": 0, "z1": 0, "pad_left": pad_left, "pad_right": pad_right}

    def apply(self, img, mode, z0, z1, pad_left, pad_right, **params):
        if mode == "none":
            return img
        if mode == "crop":
            return img[:, :, z0:z1]
        # mode == "pad"
        return np.pad(
            img,
            ((0, 0), (0, 0), (pad_left, pad_right)),
            mode="constant",
            constant_values=self.value,
        )

    def apply_to_mask(self, mask, mode, z0, z1, pad_left, pad_right, **params):
        if mode == "none":
            return mask
        if mode == "crop":
            return mask[:, :, z0:z1]
        return np.pad(
            mask,
            ((0, 0), (0, 0), (pad_left, pad_right)),
            mode="constant",
            constant_values=self.mask_value,
        )

    def get_transform_init_args_names(self):
        return ("target_z", "value", "mask_value")


class ACDC_Augmentations:
    def __init__(self, crop_transform,target_z=10):
        self.crop_transform = crop_transform
        self.target_z = target_z

    def get_training_augmentation(self):
        train_transform = [
            self.crop_transform,
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.90, 1.10),
                rotate=(-10, 10),
                shear=(-5, 5),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            ForegroundPercentileClip(background_value=0, low=1, high=99, p=1.0),

            A.GaussNoise(
                std_range=(0.0, 0.03),
                mean_range=(0.0, 0.0),
                per_channel=False,
                p=0.2
            ),
            MaskedNormalize(background_value=0, p=1.0),
            PadZTo(target_z=self.target_z, value=0, mask_value=0, p=1.0),
            A.ToTensorV2(transpose_mask=True),
        ]
        return A.Compose(train_transform, additional_targets={"image2": "image","mask2": "mask"})
    
    def get_segmentation_training_augmentation(self):
        train_transform = [
            self.crop_transform,
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.90, 1.10),
                rotate=(-10, 10),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.HorizontalFlip(p=0.3),
            A.ElasticTransform(
                alpha=1.0,
                sigma=50,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.25
            ),
            ForegroundPercentileClip(background_value=0, low=1, high=99, p=1.0),
            A.GaussNoise(
                std_range=(0.0, 0.03),
                mean_range=(0.0, 0.0),
                per_channel=False,
                p=0.2
            ),
            MaskedNormalize(background_value=0, p=1.0),
            PadZTo(target_z=self.target_z, value=0, mask_value=0, p=1.0),
            A.ToTensorV2(transpose_mask=True),
        ]
        return A.Compose(train_transform, additional_targets={"image2": "image","mask2": "mask"})

    def get_segmentation_training_augmentation_2(self):
        train_transform = [
            self.crop_transform,
            A.Affine(
                translate_percent={"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
                scale=(0.90, 1.10),
                rotate=(-10, 10),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            ForegroundPercentileClip(background_value=0, low=1, high=99, p=1.0),
            MaskedNormalize(background_value=0, p=1.0),
            PadZTo(target_z=self.target_z, value=0, mask_value=0, p=1.0),
            A.ToTensorV2(transpose_mask=True),
        ]
        return A.Compose(train_transform, additional_targets={"image2": "image","mask2": "mask"})

    def get_validation_augmentation(self):
        val_transform = [
            self.crop_transform,
            ForegroundPercentileClip(background_value=0, low=1, high=99, p=1.0),
            MaskedNormalize(background_value=0, p=1.0),
            PadZTo(target_z=self.target_z, value=0, mask_value=0, p=1.0),
            A.ToTensorV2(transpose_mask=True),
        ]
        return A.Compose(val_transform, additional_targets={"image2": "image","mask2": "mask"})
    

    def get_segmentation_training_augmentation_2d(self):
        train_transform = [
            self.crop_transform,
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.90, 1.10),
                rotate=(-10, 10),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                fill=0,
                fill_mask=0,
                p=0.5,
            ),
            A.HorizontalFlip(p=0.3),
            A.ElasticTransform(
                alpha=1.0,
                sigma=50,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.25
            ),
            ForegroundPercentileClip(background_value=0, low=1, high=99, p=1.0),
            A.GaussNoise(
                std_range=(0.0, 0.03),
                mean_range=(0.0, 0.0),
                per_channel=False,
                p=0.2
            ),
            MaskedNormalize(background_value=0, p=1.0),
            A.ToTensorV2(transpose_mask=True),
        ]
        return A.Compose(train_transform)
    
    def get_validation_augmentation_2d(self):
        val_transform = [
            self.crop_transform,
            ForegroundPercentileClip(background_value=0, low=1, high=99, p=1.0),
            MaskedNormalize(background_value=0, p=1.0),
            A.ToTensorV2(transpose_mask=True),
        ]
        return A.Compose(val_transform)
    
    def get_replay_validation_augmentation(self):
        val_transform = [
            self.crop_transform,
            ForegroundPercentileClip(background_value=0, low=1, high=99, p=1.0),
            MaskedNormalize(background_value=0, p=1.0),
            PadZTo(target_z=self.target_z, value=0, mask_value=0, p=1.0),
            A.ToTensorV2(transpose_mask=True),
        ]
        return A.ReplayCompose(val_transform, additional_targets={"image2": "image","mask2": "mask"})