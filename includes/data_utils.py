# includes/data_utils.py

import os
from typing import Tuple, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .config import TrainingConfig


def _get_data_root(cfg: TrainingConfig) -> str:
    """Return the base directory that contains train_data, test_data, train_labels.csv."""
    if os.path.isabs(cfg.data_dir):
        return cfg.data_dir
    return os.path.join(cfg.project_root, cfg.data_dir)


def build_paths(cfg: TrainingConfig) -> Tuple[str, str, str]:
    """Return absolute paths for train images, test images and labels csv."""
    base = _get_data_root(cfg)
    train_img_dir = os.path.join(base, cfg.train_img_dir)
    test_img_dir = os.path.join(base, cfg.test_img_dir)
    labels_path = os.path.join(base, cfg.labels_csv)
    return train_img_dir, test_img_dir, labels_path


def load_full_labels(
    cfg: TrainingConfig,
) -> Tuple[pd.DataFrame, List[str], Dict[str, int], Dict[int, str]]:
    """
    Load the full labels dataframe and build label encodings, without splitting.

    This is used both by the simple hold-out split and by StratifiedKFold.
    """
    train_img_dir, _, labels_path = build_paths(cfg)

    labels_df = pd.read_csv(labels_path)

    # Encode labels as integers
    unique_labels = sorted(labels_df["label"].unique())
    label_to_idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    idx_to_label = {idx: lbl for lbl, idx in label_to_idx.items()}

    labels_df["label_idx"] = labels_df["label"].map(label_to_idx)

    # Full image path (useful for debugging / Dataset)
    labels_df["img_path"] = labels_df["sample_index"].apply(
        lambda fname: os.path.join(train_img_dir, fname)
    )

    return labels_df, unique_labels, label_to_idx, idx_to_label


def load_labels_and_split(
    cfg: TrainingConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, int], Dict[int, str]]:
    """Load labels and perform a single stratified train/validation split."""
    labels_df, unique_labels, label_to_idx, idx_to_label = load_full_labels(cfg)

    # Train/val split (hold-out)
    train_df, val_df = train_test_split(
        labels_df,
        test_size=cfg.val_size,
        random_state=cfg.random_seed,
        stratify=labels_df["label_idx"],
    )

    return train_df, val_df, unique_labels, label_to_idx, idx_to_label


class DoctogresDataset(Dataset):
    """Dataset for training/validation.
    
    It can optionally use binary masks to crop the lesion region or
    multiply the image by the mask.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: str,
        mask_dir: Optional[str] = None,
        transform=None,
        use_masks: bool = False,
        mask_mode: str = "crop_bbox",   # "crop_bbox" or "multiply"
    ):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.mask_dir = mask_dir if mask_dir is not None else img_dir
        self.transform = transform
        self.use_masks = use_masks
        self.mask_mode = mask_mode

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _get_mask_bbox(mask_img: Image.Image) -> Optional[tuple]:
        """Return bounding box (left, top, right, bottom) from a binary mask.
        If mask is empty, return None.
        """
        mask_np = np.array(mask_img)  # expected 0/255
        ys, xs = np.where(mask_np > 0)

        if xs.size == 0 or ys.size == 0:
            return None

        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        margin = 5
        h, w = mask_np.shape
        x_min = max(x_min - margin, 0)
        y_min = max(y_min - margin, 0)
        x_max = min(x_max + margin, w - 1)
        y_max = min(y_max + margin, h - 1)

        # PIL bbox: (left, upper, right, lower) with right/lower exclusive
        return (x_min, y_min, x_max + 1, y_max + 1)

    def _apply_mask(
        self,
        img: Image.Image,
        img_name: str,
    ) -> Image.Image:
        """Apply the selected mask_mode to the image if mask exists."""
        mask_name = img_name.replace("img_", "mask_")
        mask_path = os.path.join(self.mask_dir, mask_name)

        if not os.path.exists(mask_path):
            return img  # no mask, return original

        mask_img = Image.open(mask_path).convert("L")

        if self.mask_mode == "crop_bbox":
            bbox = self._get_mask_bbox(mask_img)
            if bbox is not None:
                img = img.crop(bbox)
            return img

        elif self.mask_mode == "multiply":
            # Multiply image by mask: img * mask (0/1)
            # Ensure mask and image have same size
            if mask_img.size != img.size:
                mask_img = mask_img.resize(img.size, Image.NEAREST)

            img_np = np.array(img, dtype=np.float32)
            mask_np = np.array(mask_img, dtype=np.float32) / 255.0
            mask_np = np.clip(mask_np, 0.0, 1.0)

            # Expand mask to 3 channels if needed
            if mask_np.ndim == 2:
                mask_np = mask_np[..., None]
            if mask_np.shape[-1] == 1:
                mask_np = np.repeat(mask_np, 3, axis=-1)

            img_np = img_np * mask_np
            img_np = np.clip(img_np, 0.0, 255.0).astype(np.uint8)
            img = Image.fromarray(img_np)
            return img

        else:
            # Unknown mode -> do nothing
            return img

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_name = row["sample_index"]
        label = int(row["label_idx"])

        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.use_masks:
            img = self._apply_mask(img, img_name)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class DoctogresTestDataset(Dataset):
    """Dataset for test images (no labels)."""

    def __init__(self, file_list: List[str], img_dir: str, transform=None):
        self.file_list = file_list
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int):
        fname = self.file_list[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, fname


def get_transforms(cfg: TrainingConfig):
    """Return train and validation transforms based on the config."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(cfg.img_size + 32),
        transforms.CenterCrop(cfg.img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, val_transform


def create_dataloaders(
    cfg: TrainingConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    train_transform,
    val_transform,
):
    """Create train and validation dataloaders."""
    train_img_dir, _, _ = build_paths(cfg)

    base_data_dir = _get_data_root(cfg)
    if cfg.mask_dir is None:
        mask_dir = train_img_dir
    else:
        mask_dir = os.path.join(base_data_dir, cfg.mask_dir)

    train_ds = DoctogresDataset(
        train_df,
        img_dir=train_img_dir,
        mask_dir=mask_dir,
        transform=train_transform,
        use_masks=cfg.use_masks,
        mask_mode=cfg.mask_mode,
    )

    val_ds = DoctogresDataset(
        val_df,
        img_dir=train_img_dir,
        mask_dir=mask_dir,
        transform=val_transform,
        use_masks=cfg.use_masks,
        mask_mode=cfg.mask_mode,
    )


    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory,
    )

    return train_loader, val_loader
