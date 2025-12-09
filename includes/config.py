from dataclasses import dataclass
from typing import Optional, Dict, List, Any

PREPROCESSING_KEYS: List[str] = [
        "pp_remove_shrek",
        "pp_fix_stained",
        "pp_split_doubles",
        "pp_remove_black_rect",
        "pp_padding_square",
        "pp_crop_to_mask",
        "pp_resize_and_normalize",
        "pp_augmentation_mode",
        "pp_crop_padding",
        "pp_target_size",
        "pp_apply_clahe",
        "pp_clahe_clip_limit",
        "pp_clahe_tile_grid",
        "pp_num_aug_copies",
        "pp_strong_rotation_degrees",
        "pp_strong_zoom_min",
        "pp_strong_zoom_max",
        "pp_strong_brightness",
        "pp_strong_contrast",
        "pp_strong_saturation",
        "pp_strong_hue",
        "pp_strong_random_erasing_p"


]

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
# mask_mode: ["multiply", "crop_bbox"]

GRID_SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {

    "preprocessing": {
        # ===== Dataset (usa output di preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],

        # ==== PREPROCESSING CONFIG (solo UNO per chiave) ====
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_resize_and_normalize": [True],

        "pp_augmentation_mode": ["strong"],      # "none" | "basic" | "strong"

        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_apply_clahe": [False],
        "pp_clahe_clip_limit": [2.0],
        "pp_clahe_tile_grid": [(16, 16)],
        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.8],
        "pp_strong_zoom_max": [1.0],

        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],


        "execute" : False
    },
  

    "test": {
        # ===== Dataset (usa output di preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],

        # ==== PREPROCESSING CONFIG (solo UNO per chiave) ====
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_resize_and_normalize": [True],

        "pp_augmentation_mode": ["strong"],      # "none" | "basic" | "strong"

        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_apply_clahe": [False],
        "pp_clahe_clip_limit": [2.0],
        "pp_clahe_tile_grid": [(16, 16)],
        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.8],
        "pp_strong_zoom_max": [1.0],

        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],


        "execute" : True,
        # ===== Hyperparam
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

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [5],
    },


    # ----------------------------------------------------------------------
    # ResNet50 su dataset ORIGINALE, SENZA preprocessing offline
    # (usa direttamente train_data / test_data / train_labels.csv)
    # ----------------------------------------------------------------------
    
    "resnet50_img384": {
        # ===== Dataset (raw)
        "train_img_dir": ["train_data"],
        "test_img_dir": ["test_data"],
        "labels_csv": ["train_labels.csv"],

        # ===== Preprocessing offline =====

        "execute" : True,
        # ===== Hyperparam
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

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [5],
    },

    # ----------------------------------------------------------------------
    # ResNet50 su dataset PREPROCESSATO + strong augmentation offline
    # (usa pp_train_data / pp_test_data / pp_train_labels.csv)
    # ----------------------------------------------------------------------

    "resnet50_img384_preprocessing": {
        # ===== Dataset (usa output di preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],

        # ==== PREPROCESSING CONFIG (solo UNO per chiave) ====
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_resize_and_normalize": [True],

        "pp_augmentation_mode": ["strong"],      # "none" | "basic" | "strong"

        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_apply_clahe": [False],
        "pp_clahe_clip_limit": [2.0],
        "pp_clahe_tile_grid": [(16, 16)],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.8],
        "pp_strong_zoom_max": [1.0],

        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],


        "execute" : True,
        # ===== Hyperparam
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

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [5],
    },
}
