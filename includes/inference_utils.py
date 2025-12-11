# includes/inference_utils.py

import os
from typing import Dict, List, Tuple
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import TrainingConfig
from .data_utils import DoctogresTestDataset, build_paths


def create_test_loader(
    cfg: TrainingConfig,
    val_transform,
) -> Tuple[DataLoader, List[str]]:
    """Create DataLoader for test images."""
    _, test_img_dir, _ = build_paths(cfg)

    # Only take img_*.png, ignore mask_*.png
    test_files = sorted(
        f for f in os.listdir(test_img_dir)
        if f.lower().endswith(".png") and f.startswith("img_")
    )
    print("Number of test images:", len(test_files))
    print("First 5 test files:", test_files[:5])

    test_ds = DoctogresTestDataset(
        test_files,
        img_dir=test_img_dir,
        transform=val_transform,
    )

    use_pin_memory = torch.cuda.is_available()

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=use_pin_memory,
    )

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
