from dataclasses import dataclass
from typing import Optional, Dict, List, Any


@dataclass
class TrainingConfig:
    """Base config; hyperparameters will be attached during grid search."""

    # logical experiment name
    exp_name: str

    # ---- PATH & DATA ----
    project_root: str = "."
    data_dir: str = "data"
    out_dir: str = "out"

    train_img_dir: str = "train_data"
    test_img_dir: str = "test_data"
    labels_csv: str = "train_labels.csv"
    mask_dir: Optional[str] = None

    # ---- DATA SPLIT ----
    val_size: float = 0.2
    random_seed: int = 42


# cv_type: ["holdout", "kfold"]
# 


GRID_SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {
    "effb0_224_maskmul_f1": {
        "backbone": ["efficientnet_b0", "resnet50"],
        "img_size": [224, 384],
        "batch_size": [8, 16],
        "num_workers": [0, 4],
        "lr": [1e-4, 3e-4],
        "weight_decay": [1e-4, 1e-5],
        "epochs": [30, 50],
        "use_scheduler": [True],
        "use_masks": [True, False],
        "mask_mode": ["multiply", "crop_bbox"],

        # NEW: validation strategy
        # "holdout" = split singolo (come ora)
        # "kfold"   = StratifiedKFold
        "cv_type": ["holdout"],      # oppure ["holdout", "kfold"]
        "n_splits": [5],             # usato solo se cv_type == "kfold"
    },

    "resnet_only": {
        "backbone": ["resnet18", "resnet50"],
        "img_size": [224, 384],
        "batch_size": [16],
        "num_workers": [4],
        "lr": [1e-4, 3e-4],
        "weight_decay": [1e-4],
        "epochs": [30],
        "use_scheduler": [True],
        "use_masks": [False],
        "mask_mode": ["crop_bbox"],

        "cv_type": ["holdout"],
        "n_splits": [5],
    },

    "test": {
        "backbone": ["resnet18"],
        "img_size": [224],
        "batch_size": [16],
        "num_workers": [4],
        "lr": [1e-4],
        "weight_decay": [1e-4],
        "epochs": [1],
        "use_scheduler": [True],
        "use_masks": [False],
        "mask_mode": ["crop_bbox"],

        "cv_type": ["kfold"], 
        "n_splits": [5],
    },
}
