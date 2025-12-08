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

    mask_dir: Optional[str] = None

    # ---- DATA SPLIT ----
    val_size: float = 0.2
    random_seed: int = 42


# cv_type: ["holdout", "kfold"]
# augmentation: ["none", "strong"]
# mask_mode: ["multiply", "crop_bbox"],

GRID_SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {
    
    "test": {
        # Dataset originale
        "train_img_dir": ["train_data"],
        "test_img_dir": ["test_data"],
        "labels_csv": ["train_labels.csv"],

        # Hyperparam
        "backbone": ["resnet50"],
        "img_size": [128],
        "batch_size": [16],
        "num_workers": [2],
        "lr": [1e-4],
        "weight_decay": [1e-4],
        "epochs": [1],
        "use_scheduler": [True],
        "use_masks": [False],
        "mask_mode": ["multiply"],

        # Validazione
        "cv_type": ["holdout"],
        "n_splits": [5],
    },

    "resnet50_img384": {
        # Dataset originale
        "train_img_dir": ["train_data"],
        "test_img_dir": ["test_data"],
        "labels_csv": ["train_labels.csv"],

        # Hyperparam
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "num_workers": [4],
        "lr": [1e-4],
        "weight_decay": [1e-4],
        "epochs": [50],
        "use_scheduler": [True],
        "use_masks": [True],
        "mask_mode": ["multiply"],

        # Validazione
        "cv_type": ["holdout"],
        "n_splits": [5],
    },

    
    "resnet50_img384_preprocessing": {
        # Dataset originale
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],

        # Hyperparam
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "num_workers": [4],
        "lr": [1e-4],
        "weight_decay": [1e-4],
        "epochs": [1],
        "use_scheduler": [True],
        "use_masks": [True],
        "mask_mode": ["multiply"],

        # Validazione
        "cv_type": ["holdout"],
        "n_splits": [5],
    },




}