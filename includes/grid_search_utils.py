# includes/grid_search_utils.py

from __future__ import annotations

import itertools
import os
from copy import deepcopy
from dataclasses import asdict, replace
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch

from .config import TrainingConfig
from .data_utils import (
    load_labels_and_split,
    get_transforms,
    create_dataloaders,
)
from .model_utils import (
    build_model,
    create_criterion_optimizer_scheduler,
    train_model,
)


def make_exp_name(base_name: str, params: Dict[str, Any]) -> str:
    """Build a compact experiment name from base name + params."""
    parts = [base_name]
    for k, v in params.items():
        sv = str(v)
        sv = sv.replace(" ", "")
        sv = sv.replace(".", "p")
        parts.append(f"{k}-{sv}")
    return "_".join(parts)


def run_single_config(
    cfg: TrainingConfig,
    device: torch.device,
) -> Tuple[float, Dict[str, List[float]]]:
    """
    Run a full training pipeline for a single config.

    Returns:
        best_val_f1: best F1 (macro) on validation
        history: history dict from train_model
    """
    # 1) Load data and split
    train_df, val_df, unique_labels, label_to_idx, idx_to_label = load_labels_and_split(
        cfg
    )

    # 2) Transforms and dataloaders
    train_t, val_t = get_transforms(cfg)
    train_loader, val_loader = create_dataloaders(
        cfg, train_df, val_df, train_t, val_t
    )

    # 3) Model, criterion, optimizer, scheduler
    num_classes = len(unique_labels)
    model = build_model(cfg, num_classes=num_classes, device=device)

    criterion, optimizer, scheduler = create_criterion_optimizer_scheduler(
        cfg, model, train_df, device
    )

    # 4) Train model (F1 is already used inside train_model)
    best_state_dict, history = train_model(
        cfg, model, train_loader, val_loader, criterion, optimizer, scheduler, device
    )

    # 5) Extract best validation F1 from history
    val_f1_list = history.get("val_f1", [])
    best_val_f1 = max(val_f1_list) if len(val_f1_list) > 0 else 0.0

    return best_val_f1, history


def grid_search(
    base_cfg: TrainingConfig,
    param_grid: Dict[str, List[Any]],
    device: torch.device,
    save_results_path: str | None = None,
) -> pd.DataFrame:
    """
    Run a simple grid search over the given param_grid.

    Args:
        base_cfg: starting TrainingConfig (will be copied for each combo)
        param_grid: dict like {"lr": [1e-4, 3e-4], "batch_size": [8, 16], ...}
        device: torch.device
        save_results_path: optional CSV path to save the summary

    Returns:
        results_df: pandas DataFrame with one row per combination
    """
    # Pre-compute all combinations
    keys = list(param_grid.keys())
    values_list = [param_grid[k] for k in keys]
    combos = list(itertools.product(*values_list))
    n_combos = len(combos)

    print("==============================================================")
    print(f"Starting GRID SEARCH with {n_combos} combinations")
    print("Params:", keys)
    print("==============================================================")

    results: List[Dict[str, Any]] = []

    for i, values in enumerate(combos, start=1):
        params = dict(zip(keys, values))

        print("\n--------------------------------------------------------------")
        print(f"[Grid {i}/{n_combos}] params = {params}")

        # Build a new config from base_cfg + params
        cfg_i = replace(base_cfg, **params)  # dataclass replace
        cfg_i = deepcopy(cfg_i)  # safety

        # Give a specific experiment name for this combo
        cfg_i.exp_name = make_exp_name(base_cfg.exp_name, params)

        # Run training for this config
        best_val_f1, history = run_single_config(cfg_i, device)

        # Collect a flat dict with config + metrics
        cfg_dict = asdict(cfg_i)
        row: Dict[str, Any] = {}
        # Keep only a subset of config fields + tuned params, to avoid huge CSV
        keep_keys = [
            "exp_name",
            "backbone",
            "img_size",
            "batch_size",
            "lr",
            "weight_decay",
            "epochs",
            "use_masks",
            "mask_mode",
        ]
        for k in keep_keys:
            row[k] = cfg_dict.get(k)

        # Add tuned params explicitly (in case some are not in keep_keys)
        for k, v in params.items():
            row[f"gs_{k}"] = v

        row["best_val_f1"] = best_val_f1

        results.append(row)

    # Build DataFrame
    results_df = pd.DataFrame(results)

    # Sort by best F1 (descending)
    if not results_df.empty:
        results_df = results_df.sort_values("best_val_f1", ascending=False)

    # Optionally save to CSV
    if save_results_path is not None:
        os.makedirs(os.path.dirname(save_results_path), exist_ok=True)
        results_df.to_csv(save_results_path, index=False)
        print("Grid search results saved to:", save_results_path)

    print("\n================ GRID SEARCH DONE ================")
    if not results_df.empty:
        print(results_df.head())

    return results_df
