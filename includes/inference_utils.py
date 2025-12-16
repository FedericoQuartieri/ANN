# includes/inference_utils.py

import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .data_utils import DoctogresTestDataset, build_paths


def create_test_loader(
    cfg: TrainingConfig,
    val_transform,
) -> Tuple[DataLoader, List[str]]:
    """Create DataLoader for test images."""
    _, test_img_dir, _ = build_paths(cfg)
    base_data_dir = cfg.data_dir if os.path.isabs(cfg.data_dir) else os.path.join(cfg.project_root, cfg.data_dir)

    # Only take img_*.png, ignore mask_*.png
    test_files = sorted(
        f for f in os.listdir(test_img_dir)
        if f.lower().endswith(".png") and f.startswith("img_")
    )
    print("Number of test images:", len(test_files))
    print("First 5 test files:", test_files[:5])

    # Get mask directory
    if cfg.mask_dir is None:
        mask_dir = test_img_dir
    else:
        mask_dir = os.path.join(base_data_dir, cfg.mask_dir)

    # Get ROI padding
    roi_padding = getattr(cfg, 'roi_padding', 10)
    use_masks = getattr(cfg, 'use_masks', False)
    mask_mode = getattr(cfg, 'mask_mode', 'crop_bbox')

    test_ds = DoctogresTestDataset(
        test_files,
        img_dir=test_img_dir,
        mask_dir=mask_dir,
        transform=val_transform,
        use_masks=use_masks,
        mask_mode=mask_mode,
        roi_padding=roi_padding,
    )

    use_pin_memory = cfg.pin_memory and torch.cuda.is_available()
    use_persistent = bool(cfg.persistent_workers) and int(cfg.num_workers) > 0

    dl_kwargs = dict(
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=int(cfg.num_workers),
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent,
    )

    if int(cfg.num_workers) > 0:
        dl_kwargs["prefetch_factor"] = int(cfg.prefetch_factor)

    test_loader = DataLoader(test_ds, **dl_kwargs)


    return test_loader, test_files


def _get_out_root(cfg: TrainingConfig) -> str:
    """Return directory where to save submissions.

    If cfg.out_dir is absolute, use it directly.
    Otherwise, join project_root and out_dir.
    """
    if os.path.isabs(cfg.out_dir):
        return cfg.out_dir
    return os.path.join(cfg.project_root, cfg.out_dir)

def run_inference_and_save(
    cfg: TrainingConfig,
    model: torch.nn.Module,
    test_loader: DataLoader,
    idx_to_label: Dict[int, str],
    device: torch.device,
    output_csv: str | None = None,
    aggregate_tiles: bool = True,
) -> str:
    """Run inference on the test set and save a submission CSV.

    - Se output_csv è None, usa 'submission_{cfg.exp_name}.csv'
    - Altrimenti usa ESATTAMENTE il nome passato.
    - Se aggregate_tiles=True, aggrega le predizioni per le tile della stessa immagine
      usando voting pesato sulla dimensione della maschera.
    """
    model.eval()
    all_names: List[str] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)

            all_names.extend(list(names))
            all_preds.extend(preds.cpu().numpy().tolist())

    labels = [idx_to_label[i] for i in all_preds]

    # Create initial dataframe with all predictions
    raw_df = pd.DataFrame(
        {"sample_index": all_names, "label": labels}
    )

    # Aggregate tiles if requested
    if aggregate_tiles:
        submission_df = _aggregate_tiles_by_mask_weight(raw_df, cfg)
    else:
        submission_df = raw_df.sort_values("sample_index")

    # ---------- decidi il nome del file ----------
    if output_csv is None:
        exp_name = getattr(cfg, "exp_name", "experiment")
        safe_name = str(exp_name).replace(" ", "_")
        filename = f"submission_{safe_name}.csv"
    else:
        filename = output_csv

    save_dir = _get_out_root(cfg)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)

    submission_df.to_csv(out_path, index=False)
    print(f"[run_inference_and_save] exp_name={getattr(cfg, 'exp_name', None)} "
          f"| filename='{filename}'")
    print("Saved submission to:", out_path)

    # ---------- download automatico se siamo su Colab ----------
    try:
        from google.colab import files  # type: ignore
        files.download(out_path)
        print("Triggered Colab download for:", out_path)
    except Exception:
        pass

    return out_path


def _get_original_image_name(tile_name: str) -> str:
    """Extract original image name from tile name.
    
    Examples:
        img_00001.png -> img_00001.png (no tile)
        img_00001_k1.png -> img_00001.png (tile 1)
        img_00001_k2.png -> img_00001.png (tile 2)
    """
    # Remove img_ prefix and .png suffix
    base_name = tile_name.replace('img_', '').replace('.png', '')
    
    # Remove tile suffix if present (_k1, _k2, etc.)
    if '_k' in base_name:
        original_name = base_name.rsplit('_k', 1)[0]
    else:
        original_name = base_name
    
    return f"img_{original_name}.png"


def _get_mask_size(mask_path: str) -> int:
    """Calculate the number of non-zero pixels in a mask."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0
    return int(np.count_nonzero(mask > 0))


def _aggregate_tiles_by_mask_weight(
    predictions_df: pd.DataFrame,
    cfg: TrainingConfig,
) -> pd.DataFrame:
    """Aggregate tile predictions to original images using mask-weighted voting.
    
    For each original image:
    1. Find all tiles belonging to it
    2. Load mask for each tile
    3. Weight each prediction by mask size (non-zero pixels)
    4. Use weighted majority voting to determine final label
    
    Args:
        predictions_df: DataFrame with columns [sample_index, label] for all tiles
        cfg: Configuration containing paths
        
    Returns:
        DataFrame with columns [sample_index, label] for original images only
    """
    _, test_img_dir, _ = build_paths(cfg)
    
    # Group predictions by original image
    tile_groups = defaultdict(list)
    
    for _, row in predictions_df.iterrows():
        tile_name = row['sample_index']
        original_name = _get_original_image_name(tile_name)
        tile_groups[original_name].append({
            'tile_name': tile_name,
            'label': row['label']
        })
    
    print(f"\n[Tile Aggregation] Found {len(predictions_df)} tiles from {len(tile_groups)} original images")
    
    # Aggregate predictions for each original image
    final_predictions = []
    
    for original_name, tiles in sorted(tile_groups.items()):
        # Weighted votes: label -> total_weight
        label_weights = defaultdict(float)
        total_mask_size = 0
        
        for tile_info in tiles:
            tile_name = tile_info['tile_name']
            label = tile_info['label']
            
            # Get mask path (replace img_ with mask_)
            mask_name = tile_name.replace('img_', 'mask_')
            mask_path = os.path.join(test_img_dir, mask_name)
            
            # Calculate mask size (weight)
            mask_size = _get_mask_size(mask_path)
            
            # Add weighted vote
            label_weights[label] += mask_size
            total_mask_size += mask_size
        
        # Find label with maximum weight
        if label_weights:
            final_label = max(label_weights.items(), key=lambda x: x[1])[0]
        else:
            # Fallback: use first tile's label if no masks found
            final_label = tiles[0]['label']
        
        final_predictions.append({
            'sample_index': original_name,
            'label': final_label
        })
    
    # Create final dataframe
    result_df = pd.DataFrame(final_predictions).sort_values('sample_index')
    
    print(f"[Tile Aggregation] Output contains {len(result_df)} original test images")
    
    return result_df


# =============================================================================
# ENSEMBLE INFERENCE (K-Fold model averaging)
# =============================================================================

def run_ensemble_inference_and_save(
    cfg: TrainingConfig,
    fold_state_dicts: List[Dict],
    build_model_fn,  # function: (cfg, num_classes, device) -> nn.Module
    num_classes: int,
    test_loader: DataLoader,
    idx_to_label: Dict[int, str],
    device: torch.device,
    output_csv: str | None = None,
    aggregate_tiles: bool = True,
) -> str:
    """Run ensemble inference using multiple fold models and save submission.
    
    This function:
    1. Loads each fold's model weights
    2. For each sample, gets prediction from each model
    3. Uses majority voting to determine final prediction
    4. Optionally aggregates tile predictions per original image
    
    Args:
        cfg: Training configuration
        fold_state_dicts: List of state_dict from each fold
        build_model_fn: Function to build model architecture
        num_classes: Number of classes
        test_loader: DataLoader for test images
        idx_to_label: Mapping from class index to label string
        device: torch device
        output_csv: Output filename (optional)
        aggregate_tiles: Whether to aggregate tile predictions
        
    Returns:
        Path to saved submission CSV
    """
    n_folds = len(fold_state_dicts)
    print(f"\n[Ensemble Inference] Using {n_folds} fold models")
    
    # Build models and load weights
    models = []
    for i, state_dict in enumerate(fold_state_dicts):
        model = build_model_fn(cfg, num_classes=num_classes, device=device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        print(f"  Loaded fold {i+1}/{n_folds}")
    
    all_names: List[str] = []
    all_preds: List[int] = []
    
    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(device)
            batch_size = images.size(0)
            
            # Collect predictions from all models (voting)
            fold_preds = []
            
            for model in models:
                outputs = model(images)
                _, preds = torch.max(outputs, dim=1)
                fold_preds.append(preds.cpu().numpy())
            
            # Majority voting: most common prediction across folds
            fold_preds = np.array(fold_preds)  # shape: (n_folds, batch_size)
            
            # For each sample, take the most frequent prediction
            from scipy import stats
            ensemble_preds, _ = stats.mode(fold_preds, axis=0, keepdims=False)
            
            all_names.extend(list(names))
            all_preds.extend(ensemble_preds.tolist())
    
    # Clean up models
    for model in models:
        del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    labels = [idx_to_label[i] for i in all_preds]
    
    # Create initial dataframe with all predictions
    raw_df = pd.DataFrame({
        "sample_index": all_names,
        "label": labels
    })
    
    # Aggregate tiles if requested
    if aggregate_tiles:
        submission_df = _aggregate_tiles_by_mask_weight(raw_df, cfg)
    else:
        submission_df = raw_df.sort_values("sample_index")
    
    # ---------- decide filename ----------
    if output_csv is None:
        exp_name = getattr(cfg, "exp_name", "experiment")
        safe_name = str(exp_name).replace(" ", "_")
        filename = f"submission_ensemble_{safe_name}.csv"
    else:
        filename = output_csv
    
    save_dir = _get_out_root(cfg)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, filename)
    
    submission_df.to_csv(out_path, index=False)
    print(f"[Ensemble Inference] Saved ensemble submission to: {out_path}")
    
    # ---------- auto download if on Colab ----------
    try:
        from google.colab import files  # type: ignore
        files.download(out_path)
        print("Triggered Colab download for:", out_path)
    except Exception:
        pass
    
    return out_path


def _aggregate_tiles_by_probs(
    tile_probs: Dict[str, List[np.ndarray]],
    tile_weights: Dict[str, List[float]],
    idx_to_label: Dict[int, str],
) -> pd.DataFrame:
    """Aggregate tile probabilities to original images using weighted averaging.
    
    For each original image:
    1. Collect all tile probability vectors
    2. Weight each by mask size
    3. Weighted average of probabilities
    4. Argmax for final prediction
    
    Args:
        tile_probs: Dict mapping original_name -> list of probability vectors
        tile_weights: Dict mapping original_name -> list of mask weights
        idx_to_label: Mapping from class index to label
        
    Returns:
        DataFrame with [sample_index, label] for original images
    """
    final_predictions = []
    
    for original_name in sorted(tile_probs.keys()):
        probs_list = tile_probs[original_name]
        weights_list = tile_weights[original_name]
        
        # Stack and weight
        probs_array = np.array(probs_list)  # shape: (n_tiles, n_classes)
        weights_array = np.array(weights_list)  # shape: (n_tiles,)
        
        # Normalize weights
        total_weight = weights_array.sum()
        if total_weight > 0:
            weights_array = weights_array / total_weight
        else:
            # Equal weights if no mask info
            weights_array = np.ones(len(weights_list)) / len(weights_list)
        
        # Weighted average of probabilities
        avg_probs = np.average(probs_array, axis=0, weights=weights_array)
        
        # Argmax for prediction
        pred_idx = int(np.argmax(avg_probs))
        final_label = idx_to_label[pred_idx]
        
        final_predictions.append({
            'sample_index': original_name,
            'label': final_label
        })
    
    return pd.DataFrame(final_predictions).sort_values('sample_index')