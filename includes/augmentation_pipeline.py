# includes/augmentation_pipeline.py

import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as transforms
from sklearn.metrics import classification_report

from .augmented_dataset import AugmentedDataset
from .data_utils import build_paths
from .model_utils import (
    build_model,
    create_criterion_optimizer_scheduler,
    train_model,
    evaluate,
)
from .inference_utils import run_inference_and_save


def load_images_to_numpy(df, img_dir, target_size=(224, 224)):
    """
    Load images from DataFrame into a numpy array.
    
    Args:
        df: DataFrame with 'sample_index' and 'label_idx' columns
        img_dir: Directory containing images
        target_size: Tuple (height, width) for resizing
    
    Returns:
        images: numpy array of shape (N, H, W, 3) with dtype float32 in range [0, 1]
        labels: numpy array of shape (N,) with dtype int64
    """
    images = []
    labels = []
    total = len(df)
    
    print(f"Loading {total} images...")
    for i, (_, row) in enumerate(df.iterrows()):
        img_path = os.path.join(img_dir, row['sample_index'])
        
        # Load image with cv2
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not load {img_path}")
            continue
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        images.append(img)
        labels.append(row['label_idx'])

        # Lightweight progress update every 100 images
        if (i + 1) % 100 == 0 or (i + 1) == total:
            print(f"  Loaded {i+1}/{total} images")
    
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)
    
    print(f"Loaded {len(images)} images with shape {images.shape}")
    return images, labels


def create_augmentation_transforms(
    flip_p=0.5,
    rotation_degrees=15,
    color_jitter_params=None,
    random_erasing_p=0.2,
    random_erasing_scale=(0.02, 0.33),
    random_erasing_ratio=(0.3, 3.3),
):
    """
    Create training augmentation pipeline with configurable parameters.
    
    Args:
        flip_p: Probability for horizontal flip
        rotation_degrees: Max rotation angle
        color_jitter_params: Dict with brightness, contrast, saturation, hue (or None to skip)
        random_erasing_p: Probability for random erasing
        random_erasing_scale: Scale range for random erasing
        random_erasing_ratio: Aspect ratio range for random erasing
    
    Returns:
        transforms.Compose: Augmentation pipeline
    """
    aug_list = [transforms.RandomHorizontalFlip(p=flip_p)]
    
    if rotation_degrees > 0:
        aug_list.append(transforms.RandomRotation(degrees=rotation_degrees))
    
    if color_jitter_params:
        aug_list.append(transforms.ColorJitter(**color_jitter_params))
    
    if random_erasing_p > 0:
        aug_list.append(
            transforms.RandomErasing(
                p=random_erasing_p,
                scale=random_erasing_scale,
                ratio=random_erasing_ratio,
                value=0,
            )
        )
    
    return transforms.Compose(aug_list)


def prepare_augmented_data(cfg, train_df, val_df):
    """
    Load images from train_df and val_df into numpy arrays.
    
    Returns:
        X_train, y_train, X_val, y_val
    """
    train_img_dir, _, _ = build_paths(cfg)
    
    X_train, y_train = load_images_to_numpy(
        train_df, train_img_dir, target_size=(cfg.img_size, cfg.img_size)
    )
    X_val, y_val = load_images_to_numpy(
        val_df, train_img_dir, target_size=(cfg.img_size, cfg.img_size)
    )
    
    print(f"\nTrain set: {X_train.shape}, labels: {y_train.shape}")
    print(f"Val set: {X_val.shape}, labels: {y_val.shape}")
    
    return X_train, y_train, X_val, y_val


def create_augmented_loaders(
    X_train,
    y_train,
    X_val,
    y_val,
    batch_size,
    train_transform=None,
    num_workers=0,
    use_cuda=False,
):
    """
    Create augmented datasets and DataLoaders.
    
    Args:
        X_train, y_train: Training images and labels (numpy arrays)
        X_val, y_val: Validation images and labels (numpy arrays)
        batch_size: Batch size for DataLoaders
        train_transform: Transform to apply to training data
        num_workers: Number of workers for DataLoader
        use_cuda: Whether to pin memory for CUDA
    
    Returns:
        train_aug_loader, val_aug_loader
    """
    train_aug_ds = AugmentedDataset(X_train, y_train, transform=train_transform)
    val_aug_ds = AugmentedDataset(X_val, y_val, transform=None)
    
    print(f"Train augmented dataset: {len(train_aug_ds)} samples")
    print(f"Val augmented dataset: {len(val_aug_ds)} samples")
    
    train_aug_loader = DataLoader(
        train_aug_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    
    val_aug_loader = DataLoader(
        val_aug_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    
    print(f"\nCreated DataLoaders:")
    print(f"  Train batches: {len(train_aug_loader)}")
    print(f"  Val batches: {len(val_aug_loader)}")
    print(f"  Batch size: {batch_size}")
    
    return train_aug_loader, val_aug_loader


def train_with_augmentation(
    cfg,
    train_aug_loader,
    val_aug_loader,
    unique_labels,
    train_df,
    device,
):
    """
    Train a model with augmented data loaders.
    
    Returns:
        aug_model (trained), aug_history
    """
    print("\n" + "=" * 60)
    print("Training with Augmented Data")
    print("=" * 60)
    
    aug_model = build_model(cfg, num_classes=len(unique_labels), device=device)
    aug_criterion, aug_optimizer, aug_scheduler = create_criterion_optimizer_scheduler(
        cfg, model=aug_model, train_df=train_df, device=device
    )
    
    print(f"Train batches: {len(train_aug_loader)}, Val batches: {len(val_aug_loader)}")
    
    best_aug_state, aug_history = train_model(
        cfg,
        aug_model,
        train_aug_loader,
        val_aug_loader,
        aug_criterion,
        aug_optimizer,
        aug_scheduler,
        device,
    )
    
    aug_model.load_state_dict(best_aug_state)
    
    # Final evaluation
    val_loss, val_acc, y_true, y_pred = evaluate(
        aug_model, val_aug_loader, aug_criterion, device
    )
    
    print(f"\n{'=' * 60}")
    print(f"Augmented Training Results:")
    print(f"{'=' * 60}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=unique_labels))
    
    return aug_model, aug_history


def run_augmented_experiment(
    cfg,
    train_df,
    val_df,
    unique_labels,
    idx_to_label,
    device,
    val_t,
    # Augmentation parameters
    flip_p=0.5,
    rotation_degrees=15,
    use_color_jitter=True,
    random_erasing_p=0.2,
    # Training parameters
    num_workers=0,
    # Output
    save_submission=True,
    submission_name="submission_augmented.csv",
):
    """
    Run complete augmented training experiment with configurable parameters.
    
    Args:
        cfg: TrainingConfig
        train_df, val_df: DataFrames with training/validation data
        unique_labels: List of unique label names
        idx_to_label: Dict mapping label index to name
        device: torch.device
        val_t: Validation transforms
        flip_p: Probability for horizontal flip
        rotation_degrees: Max rotation angle
        use_color_jitter: Whether to use color jitter
        random_erasing_p: Probability for random erasing
        num_workers: Number of DataLoader workers (0 for Windows notebooks)
        save_submission: Whether to save submission CSV
        submission_name: Name for submission CSV file
    
    Returns:
        aug_model, aug_history
    """
    # 1. Load images into numpy arrays
    print("\n" + "=" * 60)
    print("STEP 1: Loading images into memory")
    print("=" * 60)
    X_train, y_train, X_val, y_val = prepare_augmented_data(cfg, train_df, val_df)
    
    # 2. Create augmentation transforms
    print("\n" + "=" * 60)
    print("STEP 2: Creating augmentation pipeline")
    print("=" * 60)
    color_jitter_params = (
        {"brightness": 0.2, "contrast": 0.2, "saturation": 0.2, "hue": 0.1}
        if use_color_jitter
        else None
    )
    train_transform = create_augmentation_transforms(
        flip_p=flip_p,
        rotation_degrees=rotation_degrees,
        color_jitter_params=color_jitter_params,
        random_erasing_p=random_erasing_p,
    )
    print("Augmentation transforms:")
    print(train_transform)
    
    # 3. Create augmented DataLoaders
    print("\n" + "=" * 60)
    print("STEP 3: Creating augmented DataLoaders")
    print("=" * 60)
    train_aug_loader, val_aug_loader = create_augmented_loaders(
        X_train,
        y_train,
        X_val,
        y_val,
        batch_size=cfg.batch_size,
        train_transform=train_transform,
        num_workers=num_workers,
        use_cuda=torch.cuda.is_available(),
    )
    
    # 4. Train model with augmentation
    print("\n" + "=" * 60)
    print("STEP 4: Training model")
    print("=" * 60)
    aug_model, aug_history = train_with_augmentation(
        cfg, train_aug_loader, val_aug_loader, unique_labels, train_df, device
    )
    
    # 5. Save submission (optional)
    if save_submission:
        print("\n" + "=" * 60)
        print("STEP 5: Generating submission")
        print("=" * 60)
        from .inference_utils import create_test_loader
        
        test_loader, test_files = create_test_loader(cfg, val_t)
        run_inference_and_save(
            cfg, aug_model, test_loader, idx_to_label, device, output_csv=submission_name
        )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    
    return aug_model, aug_history
