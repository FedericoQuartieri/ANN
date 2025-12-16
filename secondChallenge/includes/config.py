from dataclasses import dataclass
from typing import Optional, Dict, List, Any

PREPROCESSING_KEYS: List[str] = [
        "pp_remove_shrek",
        "pp_fix_stained",
        "pp_split_doubles",
        "pp_remove_black_rect",
        "pp_padding_square",
        "pp_crop_to_mask",
        "pp_split_into_tiles",
        "pp_remove_empty_masks",
        "pp_darken_outside_mask",
        "pp_augmentation_enabled",
        "pp_target_size",
        "pp_crop_padding",
        "pp_smart_discard_threshold",
        "pp_num_aug_copies",
        "pp_strong_rotation_degrees",
        "pp_strong_zoom_min",
        "pp_strong_zoom_max",
        "pp_strong_brightness",
        "pp_strong_contrast",
        "pp_strong_saturation",
        "pp_strong_hue",
        "pp_strong_random_erasing_p",
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
    random_seed: int = 42
    
    # ---- ROI PARAMETERS ----
    use_roi_crop: bool = True      # Use ROI crop strategy (recommended)
    roi_padding: int = 10          # Padding in pixels for ROI crop (10-20 recommended)
    num_workers: int = 0
    pin_memory: bool = True
    persistent_workers: bool = False
    prefetch_factor: int = 2
    
    # ---- REGULARIZATION ----
    dropout_rate: float = 0.0      # Dropout rate for classifier head (0.0-0.5)
    label_smoothing: float = 0.0   # Label smoothing for CrossEntropyLoss (0.0-0.2)
    
    # ---- EARLY STOPPING ----
    early_stopping: bool = False   # Enable early stopping
    early_stopping_patience: int = 5  # Stop after N epochs without improvement
    early_stopping_min_delta: float = 0.001  # Minimum improvement to reset patience


# cv_type: ["holdout", "kfold"]
# mask_mode: ["multiply", "crop_bbox"]
# backbone: [resnet50, resnet18, efficientnet_b0, convnext_tiny, efficientnet_b3]

GRID_SEARCH_SPACES: Dict[str, Dict[str, List[Any]]] = {

    "preprocessing": {
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],

        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [True],
        "pp_crop_to_mask": [True],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False], # lasciare False se use_roi_crop = True
        "pp_augmentation_enabled": [True],
        
        # ROI Strategy (nuova implementazione)
        "use_roi_crop": [True],   # Usa ROI crop quadrato con padding
        "roi_padding": [10],       # Padding in pixels per ROI (10-20 consigliato)

        "pp_target_size": [256],
        "pp_crop_padding": [10],
        "pp_smart_discard_threshold": [0.05],
        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.8],
        "pp_strong_zoom_max": [1.0],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],


        "execute" : [False]
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

        "pp_augmentation_enabled": [False],

        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.8],
        "pp_strong_zoom_max": [1.0],

        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [False],
        "pp_remove_empty_masks": [False],
        "pp_darken_outside_mask": [False], # lasciare False se use_roi_crop = True
        
        # ROI Strategy
        "use_roi_crop": [True],
        "roi_padding": [10],

        "execute" : [True],
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
        "mask_mode": ["crop_bbox"],
        "use_amp": [True],  # Mixed-precision training (float16)

        # ===== Validation
        "cv_type": ["kfold"],
        "n_splits": [2],
        "val_size": [0.2],
    },

    # ----------------------------------------------------------------------
    # ResNet50 su dataset ORIGINALE, SENZA preprocessing offline
    # (usa direttamente train_data / test_data / train_labels.csv)
    # ----------------------------------------------------------------------
    
    # 0.1865
    #not ROI updated
    "resnet50_img384": {
        # ===== Dataset (raw)
        "train_img_dir": ["train_data"],
        "test_img_dir": ["test_data"],
        "labels_csv": ["train_labels.csv"],

        # ===== Preprocessing offline =====

        "execute" : [True],
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
        "use_amp": [True],  # Mixed-precision training (float16)

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },

    # ----------------------------------------------------------------------
    # ResNet50 su dataset PREPROCESSATO + strong augmentation offline
    # (usa pp_train_data / pp_test_data / pp_train_labels.csv)
    # ----------------------------------------------------------------------

    # test: 0.0246
    #not ROI updated
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

        "pp_augmentation_enabled": [False],

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


        "execute" : [True],
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
        "use_amp": [True],  # Mixed-precision training (float16)

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },

    #f1=0.1837
    #not ROI updated
    "resnet50_img384_preprocessing_augementation_test": {
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

        # >>> strong augmentation OFFLINE abilitata <<<
        "pp_augmentation_enabled": [True],

        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_apply_clahe": [False],
        "pp_clahe_clip_limit": [2.0],
        "pp_clahe_tile_grid": [(16, 16)],

        # 1 sola copia augmentata per immagine (il bilanciamento di classe lo farei a livello di sampler)
        "pp_num_aug_copies": [1],

        # --- Strong aug parametri "safe" per istologia ---
        # Rotazione random moderata (±15° circa, dipende da come lo usi nel transform)
        "pp_strong_rotation_degrees": [15],

        # Zoom lieve intorno a 1.0 (niente crop estremi)
        "pp_strong_zoom_min": [0.9],
        "pp_strong_zoom_max": [1.1],

        # Jitter di colore moderato (simula vari staining / scanner)
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],

        # Random erasing con probabilita bassa (buchi piccoli, regolarizzazione)
        "pp_strong_random_erasing_p": [0.1],

        # ===== Esecuzione =====
        "execute": [True],

        # ===== Hyperparam training =====
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
        "use_amp": [True],  # Mixed-precision training (float16)

        # ===== Validation =====
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },

    #f1=0.1837
    #not ROI updated
    "resnet50_img384_preprocessing_augementation_10_copies": {
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

        # >>> strong augmentation OFFLINE abilitata <<<
        "pp_augmentation_enabled": [True],

        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_apply_clahe": [False],
        "pp_clahe_clip_limit": [2.0],
        "pp_clahe_tile_grid": [(16, 16)],

        # 1 sola copia augmentata per immagine (il bilanciamento di classe lo farei a livello di sampler)
        "pp_num_aug_copies": [10],

        # --- Strong aug parametri "safe" per istologia ---
        # Rotazione random moderata (±15° circa, dipende da come lo usi nel transform)
        "pp_strong_rotation_degrees": [15],

        # Zoom lieve intorno a 1.0 (niente crop estremi)
        "pp_strong_zoom_min": [0.9],
        "pp_strong_zoom_max": [1.1],

        # Jitter di colore moderato (simula vari staining / scanner)
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],

        # Random erasing con probabilita bassa (buchi piccoli, regolarizzazione)
        "pp_strong_random_erasing_p": [0.1],

        # ===== Esecuzione =====
        "execute": [True],

        # ===== Hyperparam training =====
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
        "use_amp": [True],  # Mixed-precision training (float16)

        # ===== Validation =====
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },

    #f1 = 0.1167
    #not ROI updated
    "resnet50_kfold_sweep": {
        # ===== Dataset (usa output di preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],

        # ===== Preprocessing offline: solo cleaning / resize =====
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_resize_and_normalize": [True],

        # IMPORTANT: offline augmentation disabilitata
        "pp_augmentation_enabled": [False],

        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_apply_clahe": [False],
        "pp_clahe_clip_limit": [2.0],
        "pp_clahe_tile_grid": [(16, 16)],

        # Ignorati quando pp_augmentation_enabled = False
        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.9],
        "pp_strong_zoom_max": [1.1],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # ===== Esecuzione =====
        "execute": [True],

        # ===== Hyperparameters (GRID) =====
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "num_workers": [4],
        "lr": [1e-4],                    # fisso, così non esplode il numero di combo

        # SWEEP su weight_decay (3 valori) e mask_mode (2 valori)
        "weight_decay": [5e-4],
        "epochs": [50],
        "use_scheduler": [True],

        "use_masks": [True],             # sempre usa le maschere
        "mask_mode": ["multiply"],   # 2 opzioni -> 6 combo totali
        "use_amp": [True],  # Mixed-precision training (float16)

        # ===== Validation: 5-fold CV =====
        "cv_type": ["kfold"],
        "n_splits": [5],
        "val_size": [0.2],               # ignorata per kfold, la lasciamo per compatibilità
    },

    #f1 = 0.2578
    #not ROI updated
    "resnet50_kfold_cropbbox_30ep": {
        # ===== Dataset (usa output di preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],

        # ===== Preprocessing offline: solo cleaning / resize =====
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_resize_and_normalize": [True],

        # NO augmentation offline
        "pp_augmentation_enabled": [False],

        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_apply_clahe": [False],
        "pp_clahe_clip_limit": [2.0],
        "pp_clahe_tile_grid": [(16, 16)],

        # Ignorati perché pp_augmentation_enabled = False
        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.9],
        "pp_strong_zoom_max": [1.1],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # ===== Esecuzione =====
        "execute": True,

        # ===== Hyperparameters (UNA SOLA CONFIG) =====
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "num_workers": [4],
        "lr": [3e-4],
        "weight_decay": [1e-3],
        "epochs": [30],              # <<< 30 epochs
        "use_scheduler": [True],

        "use_masks": [True],
        "mask_mode": ["crop_bbox"],  # fisso
        "use_amp": [True],  # Mixed-precision training (float16)

        # ===== Validation: 5-fold CV =====
        "cv_type": ["kfold"],
        "n_splits": [5],
        "val_size": [0.2],           # ignorata per kfold, ok
    },

    # Configurazione ottimizzata basata su resnet50_kfold_cropbbox_30ep (F1=0.2578)
    # Differenze: holdout invece di kfold + mixed-precision + batch_size aumentato
    "resnet50_cropbbox_30ep_holdout_amp": {
        # ===== Dataset (usa output di preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],

        # ===== Preprocessing offline: solo cleaning / resize =====
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_resize_and_normalize": [True],

        # NO augmentation offline
        "pp_augmentation_enabled": [False],

        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_apply_clahe": [False],
        "pp_clahe_clip_limit": [2.0],
        "pp_clahe_tile_grid": [(16, 16)],

        # Ignorati perché pp_augmentation_enabled = False
        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.9],
        "pp_strong_zoom_max": [1.1],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # ===== Esecuzione =====
        "execute": True,

        # ===== Hyperparameters =====
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [24],          # Aumentato da 16 grazie a mixed-precision
        "num_workers": [6],           # Aumentato per saturare GPU
        "lr": [3e-4],                 # Same as best kfold config
        "weight_decay": [1e-3],       # Same as best kfold config
        "epochs": [30],               
        "use_scheduler": [True],
        "label_smoothing": [0.05],    # Label smoothing leggero per ridurre overconfidence

        "use_masks": [True],
        "mask_mode": ["crop_bbox"],   # Best performing mode (da kfold test)
        "use_amp": [True],            # Mixed-precision training (2-3x più veloce)

        # ===== Validation =====
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },

    # 0.2603
    #not ROI updated
    "resnet50_img384_pp_crop_bbox_offaug4_strong": {
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

        # >>> offline augmentation più forte <<<
        "pp_augmentation_enabled": [True],
        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_apply_clahe": [False],
        "pp_clahe_clip_limit": [2.0],
        "pp_clahe_tile_grid": [(16, 16)],

        # più copie augmentate -> dataset 5x circa
        "pp_num_aug_copies": [4],

        # aug un filo più aggressiva
        "pp_strong_rotation_degrees": [20],
        "pp_strong_zoom_min": [0.85],
        "pp_strong_zoom_max": [1.15],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        "execute": [True],

        # ===== Hyperparam training =====
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "num_workers": [4],
        "lr": [1e-4],          # per ora lascerei invariato
        "weight_decay": [1e-4],
        "epochs": [50],
        "use_scheduler": [True],
        "use_masks": [True],
        "mask_mode": ["crop_bbox"],

        # ===== Validation =====
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },

    #0.2690
    #not ROI updated
    "resnet50_img384_pp_crop_bbox_offaug2_soft": {
        # ===== Dataset (preprocessing output) =====
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],

        # ===== PREPROCESSING CONFIG =====
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_resize_and_normalize": [True],

        # offline augmentation più soft
        "pp_augmentation_enabled": [True],
        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_apply_clahe": [False],
        "pp_clahe_clip_limit": [2.0],
        "pp_clahe_tile_grid": [(16, 16)],

        # solo 2 copie augmentate
        "pp_num_aug_copies": [2],

        # aug meno aggressiva
        "pp_strong_rotation_degrees": [10],
        "pp_strong_zoom_min": [0.9],
        "pp_strong_zoom_max": [1.1],
        "pp_strong_brightness": [0.15],
        "pp_strong_contrast": [0.15],
        "pp_strong_saturation": [0.15],
        "pp_strong_hue": [0.03],
        "pp_strong_random_erasing_p": [0.05],

        # ===== Esecuzione preprocessing =====
        "execute": [True],

        # ===== Hyperparam training =====
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "num_workers": [4],
        "lr": [1e-4],
        "weight_decay": [5e-4],    # più regolarizzazione
        "epochs": [40],            # leggermente meno epoch
        "use_scheduler": [True],
        "use_masks": [True],
        "mask_mode": ["crop_bbox"],

        # ===== Validation =====
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
        # appena sistemi il codice, qui metti "train_data"
    },

    # ----- changes in preprocessing ------
    # removed
    #       - pp_resize_and_normalize
    #       - pp_apply_clahe
    #       - pp_clahe_clip_limit
    #       - pp_clahe_tile_grid
    # added:
    #       - pp_smart_discard_threshold
    #       - pp_split_into_tiles
    #       - pp_remove_empty_masks
    #       - pp_darken_outside_mask


    # f1 = 0.3200
    #best val f1 = 0.41044344990450576
    "resnet50_new_preprocessing": {
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

        "pp_augmentation_enabled": [False],

        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.8],
        "pp_strong_zoom_max": [1.0],

        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False], # lasciare False se use_roi_crop = True
        
        "execute" : [True],
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
        "mask_mode": ["crop_bbox"],
        "use_amp": [True],

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },


    # Best val F1: 0.43062791493768726
    # test F1: 0.2642
    #not ROI updated
    "resnet50_new_preprocessing_noamp": {
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

        "pp_augmentation_enabled": [False],

        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.8],
        "pp_strong_zoom_max": [1.0],

        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False], # lasciare False se use_roi_crop = True
    

        "execute" : [True],
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
        "mask_mode": ["crop_bbox"],
        "use_amp": [False],

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },


    # Grid search per ottimizzare lr e weight_decay
    # Best config da provare dopo sweep: lr=1e-4, wd=1e-3 o 5e-3
    #not ROI updated
    "resnet50_kfold_50ep_gridsearch": {
        # ===== Dataset (preprocessing già fatto, NO augmentation offline)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],

        # ===== Preprocessing: solo cleaning, NO aug offline
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_resize_and_normalize": [True],

        "pp_augmentation_enabled": [False],

        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_apply_clahe": [False],
        "pp_clahe_clip_limit": [2.0],
        "pp_clahe_tile_grid": [(16, 16)],

        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.9],
        "pp_strong_zoom_max": [1.1],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # NO splitting in tiles (mantieni immagini intere)
        "pp_split_into_tiles": [False],
        "pp_remove_empty_masks": [False],
        "pp_darken_outside_mask": [False], # lasciare False se use_roi_crop = True
        "pp_smart_discard_threshold": [0.05],

        # ===== Esecuzione =====
        "execute": [True],

        # ===== GRID SEARCH: lr × weight_decay =====
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "num_workers": [4],
        
        # Sweep su learning rate (3 valori)
        "lr": [5e-5, 1e-4, 3e-4],
        
        # Sweep su weight_decay (2 valori) -> 3×2 = 6 configurazioni
        "weight_decay": [1e-3, 5e-3],
        
        "epochs": [50],
        "use_scheduler": [True],
        "use_masks": [True],
        "mask_mode": ["crop_bbox"],
        "use_amp": [True],

        # ===== 5-fold CV per robustezza =====
        "cv_type": ["kfold"],
        "n_splits": [5],
        "val_size": [0.2],
    },


    ## ----------  from now on validation set is not augmented  ----------

    # Best val F1: 0.3870366453519207
    # test F1: 0.2933
    #not ROI updated
    "resnet50_new_preprocessing_new_validation": {
        # ===== Dataset (usa output di preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        "mask_mode": ["crop_bbox"], # ignored
        "use_masks": [True], # ignored



        # ==== PREPROCESSING CONFIG ====
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],


        # == AUGMENTATION ==
        "pp_augmentation_enabled": [False],

        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.8],
        "pp_strong_zoom_max": [1.0],

        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False], # lasciare False se use_roi_crop = True
        
        # ROI Strategy
        "use_roi_crop": [True],
        "roi_padding": [10],

        "execute" : [True],
        # ===== Hyperparam
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "num_workers": [4],
        "lr": [1e-4],
        "weight_decay": [1e-4],
        "epochs": [50],
        "use_scheduler": [True],
        "use_amp": [True],

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },


    # test F1: 0.3686
    "best" :  {
        'train_img_dir': ['pp_train_data'],
        'test_img_dir': ['pp_test_data'],
        'labels_csv': ['pp_train_labels.csv'],
        'mask_mode': ['crop_bbox'],
        'use_masks': [True],
        'pp_remove_shrek': [True],
        'pp_fix_stained': [True],
        'pp_split_doubles': [True],
        'pp_remove_black_rect': [True],
        'pp_padding_square': [False],
        'pp_crop_to_mask': [False],
        'pp_augmentation_enabled': [True],
        'pp_crop_padding': [10],
        'pp_target_size': [384],
        'pp_strong_rotation_degrees': [15],
        'pp_strong_zoom_min': [0.8],
        'pp_strong_zoom_max': [1.0],
        'pp_strong_brightness': [0.2],
        'pp_strong_contrast': [0.2],
        'pp_strong_saturation': [0.2],
        'pp_strong_hue': [0.05],
        'pp_strong_random_erasing_p': [0.1],
        'pp_smart_discard_threshold': [0.02],
        'pp_split_into_tiles': [True],
        'pp_remove_empty_masks': [True],
        'pp_darken_outside_mask': [False],
        'use_roi_crop': [True],
        'roi_padding': [10],
        'execute': [True],
        'backbone': ['resnet50'],
        'img_size': [384],
        'batch_size': [16],
        'num_workers': [4],
        'lr': [0.0001],
        'weight_decay': [0.0001],
        'epochs': [20],
        'use_scheduler': [True],
        'use_amp': [True],
        'cv_type': ['holdout'],
        'n_splits': [5],
        'val_size': [0.2]
    },

    # --------- ADDED NUMWORKERS------------

    #k1 Fold best F1: 0.4638
    #k2 Fold best F1: 0.5244
    #k3 Fold best F1: 0.4617
    #k4 Fold best F1: 0.4179
    # Mean F1 over 4 folds: 0.4669
    # test F1: 0.3303
    "resnet50_new_preprocessing_new_validation_kfold": {
        # ===== Dataset (usa output di preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        "mask_mode": ["crop_bbox"], # ignored
        "use_masks": [True], # ignored



        # ==== PREPROCESSING CONFIG ====
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],


        # == AUGMENTATION ==
        "pp_augmentation_enabled": [False],

        "pp_crop_padding": [10],
        "pp_target_size": [384],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.8],
        "pp_strong_zoom_max": [1.0],

        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False], # lasciare False se use_roi_crop = True
        
        # ROI Strategy
        "use_roi_crop": [True],
        "roi_padding": [10],

        # workers
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],

        "execute" : [True],
        # ===== Hyperparam
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "num_workers": [4],
        "lr": [1e-4],
        "weight_decay": [1e-4],
        "epochs": [50],
        "use_scheduler": [True],
        "use_amp": [True],

        # ===== Validation
        "cv_type": ["kfold"],
        "n_splits": [4],
        "val_size": [0.2],
    },



    # F1 val: 0.3962
    # test F1: 0.3701
    "testing_with_persistent": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.8],
        "pp_strong_zoom_max": [1.0],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [10],


        # Workers config
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "lr": [1e-4],
        "weight_decay": [1e-4],
        "epochs": [20],
        "use_scheduler": [True],
        "use_amp": [True],

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },


    # F1 val 0.3971
    # test F1: 0.3298
    "testing_with_persistent_numcopies4": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [4],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.8],
        "pp_strong_zoom_max": [1.0],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [10],


        # Workers config
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "lr": [1e-4],
        "weight_decay": [1e-4],
        "epochs": [20],
        "use_scheduler": [True],
        "use_amp": [True],

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },

    # F1 val: 0.3502
    # test F1: 0.3507
    "testing_with_persistent_numcopies2": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [2],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.8],
        "pp_strong_zoom_max": [1.0],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [10],


        # Workers config
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "lr": [1e-4],
        "weight_decay": [1e-4],
        "epochs": [20],
        "use_scheduler": [True],
        "use_amp": [True],

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },


    # Mean F1 over 4 folds: 0.7266 (batch size: 16)
    # Mean F1 over 4 folds: 0.6104
    # test F1: 0.3669
    "resnet50_strongaug_424_batch_grid": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [424],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [4],
        "pp_strong_rotation_degrees": [45],
        "pp_strong_zoom_min": [1.0],
        "pp_strong_zoom_max": [1.5],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [10],


        # Workers config
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["resnet50"],
        "img_size": [424],
        "batch_size": [16, 8],
        "lr": [1e-4],
        "weight_decay": [1e-4],
        "epochs": [50],
        "use_scheduler": [True],
        "use_masks": [True],
        "mask_mode": ["crop_bbox"],
        "use_amp": [True],

        # ===== Validation
        "cv_type": ["kfold"],
        "n_splits": [4],
        "val_size": [0.2],
    },

    
    # 1config: Mean F1 over 4 folds: 0.7154 ()
    # 2config: Mean F1 over 4 folds: 0.7260 (wd: 1e-3)
    # Test F1: 0.3580
    "resnet50_strongaug_424_wd_grid": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [424],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [4],
        "pp_strong_rotation_degrees": [45],
        "pp_strong_zoom_min": [1.0],
        "pp_strong_zoom_max": [1.5],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [10],


        # Workers config
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["resnet50"],
        "img_size": [424],
        "batch_size": [16],
        "lr": [1e-4],
        "weight_decay": [1e-3, 5e-3],
        "epochs": [50],
        "use_scheduler": [True],
        "use_amp": [True],

        # ===== Validation
        "cv_type": ["kfold"],
        "n_splits": [4],
        "val_size": [0.2],
    },




    # F1 validation
    # F1 test: 0.3661
    "testing_with_persistent_numcopies1_batch32": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.8],
        "pp_strong_zoom_max": [1.0],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [10],


        # Workers config
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [32],
        "lr": [1e-4],
        "weight_decay": [1e-4],
        "epochs": [20],
        "use_scheduler": [True],
        "use_amp": [True],

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },

    #test f1 = 0.3243
    #Best config (from grid):
    # {'exp_name': 'testing_with_persistent_numcopies2_batch32', 'project_root': 'c:\\Users\\danie\\ANN', 'data_dir': 'data', 'out_dir': 'out', 'mask_dir': None, 'random_seed': 42, 'use_roi_crop': True, 'roi_padding': 10, 'num_workers': 2, 'pin_memory': True, 'persistent_workers': True, 'prefetch_factor': 2, 'train_img_dir': 'pp_train_data', 'test_img_dir': 'pp_test_data', 'labels_csv': 'pp_train_labels.csv', 'use_masks': True, 'mask_mode': 'crop_bbox', 'backbone': 'resnet50', 'img_size': 384, 'batch_size': 32, 'lr': 0.0001, 'weight_decay': 0.0001, 'epochs': 20, 'use_scheduler': True, 'use_amp': True, 'cv_type': 'holdout', 'n_splits': 5, 'val_size': 0.2}
    # Best val F1: 0.426482368881585
    "testing_with_persistent_numcopies2_batch32": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [2],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.8],
        "pp_strong_zoom_max": [1.0],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [10],


        # Workers config
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [32],
        "lr": [1e-4],
        "weight_decay": [1e-4],
        "epochs": [20],
        "use_scheduler": [True],
        "use_amp": [True],

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },




    ## ---CHANGES-----
    ## removed center if roi cropped is used ------
    ## added droput and label smoothing
    ## added early stopping
    ## added fold ensamble

    # val F1: 0.3712
    # Test F1: 0.3340
    "resnet50_strongaug_384": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [4],
        "pp_strong_rotation_degrees": [45],
        "pp_strong_zoom_min": [1.0],
        "pp_strong_zoom_max": [1.5],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [10],


        # Workers config
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "lr": [1e-4],
        "weight_decay": [1e-3],
        "epochs": [50],
        "use_scheduler": [True],
        "use_masks": [True],

            
        # Extra Regularization (Non presenti nello stile originale, ma necessari qui)
        "dropout_rate": [0.4],
        "label_smoothing": [0.1],

        # Early Stopping
        "early_stopping": [True],   # Enable early stopping
        "early_stopping_patience": [5],  # Stop after N epochs without improvement
        "early_stopping_min_delta": [0.001],  # Minimum improvement to reset patience

        # ===== Validation
        "cv_type": ["kfold"],
        "n_splits": [4],
        "val_size": [0.2],
    },


    ## -----  NEW VALIDATION -----

    # F1 validation: 0.4002
    # test F1: 0.3865
    "resnet50_strongaug_384_new": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [4],
        "pp_strong_rotation_degrees": [45],
        "pp_strong_zoom_min": [1.0],
        "pp_strong_zoom_max": [1.5],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [10],


        # Workers config
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "lr": [1e-4],
        "weight_decay": [1e-3],
        "epochs": [50],
        "use_scheduler": [True],
        "use_masks": [True],

            
        # Extra Regularization (Non presenti nello stile originale, ma necessari qui)
        "dropout_rate": [0.4],
        "label_smoothing": [0.1],

        # Early Stopping
        "early_stopping": [True],   # Enable early stopping
        "early_stopping_patience": [5],  # Stop after N epochs without improvement
        "early_stopping_min_delta": [0.001],  # Minimum improvement to reset patience

        # ===== Validation
        "cv_type": ["kfold"],
        "n_splits": [4],
        "val_size": [0.2],
    },

    # F1 validation: 
    # F1 test: 0.3420
    "resnet50_strongaug_384_roi10": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [45],
        "pp_strong_zoom_min": [1.0],
        "pp_strong_zoom_max": [1.5],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [10],


        # Workers config
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "lr": [1e-4],
        "weight_decay": [1e-3],
        "epochs": [50],
        "use_scheduler": [True],
        "use_masks": [True],

            
        # Extra Regularization (Non presenti nello stile originale, ma necessari qui)
        "dropout_rate": [0.2],
        "label_smoothing": [0.1],

        # Early Stopping
        "early_stopping": [True],   # Enable early stopping
        "early_stopping_patience": [5],  # Stop after N epochs without improvement
        "early_stopping_min_delta": [0.001],  # Minimum improvement to reset patience

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [4],
        "val_size": [0.2],
    },

    # F1 val: 0.44
    # F1 test: 0.3605
    "resnet50_strongaug_384_roi20": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [45],
        "pp_strong_zoom_min": [1.0],
        "pp_strong_zoom_max": [1.5],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [20],


        # Workers config
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "lr": [1e-4],
        "weight_decay": [1e-3],
        "epochs": [50],
        "use_scheduler": [True],
        "use_masks": [True],

            
        # Extra Regularization (Non presenti nello stile originale, ma necessari qui)
        "dropout_rate": [0.2],
        "label_smoothing": [0.1],

        # Early Stopping
        "early_stopping": [True],   # Enable early stopping
        "early_stopping_patience": [5],  # Stop after N epochs without improvement
        "early_stopping_min_delta": [0.001],  # Minimum improvement to reset patience

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [4],
        "val_size": [0.2],
    },

    # F1 val: 0.44
    # F1 test: 0.3696
    "resnet50_strongaug_384_roi30": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [45],
        "pp_strong_zoom_min": [1.0],
        "pp_strong_zoom_max": [1.5],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [30],


        # Workers config
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "lr": [1e-4],
        "weight_decay": [1e-3],
        "epochs": [50],
        "use_scheduler": [True],
        "use_masks": [True],

            
        # Extra Regularization (Non presenti nello stile originale, ma necessari qui)
        "dropout_rate": [0.2],
        "label_smoothing": [0.1],

        # Early Stopping
        "early_stopping": [True],   # Enable early stopping
        "early_stopping_patience": [5],  # Stop after N epochs without improvement
        "early_stopping_min_delta": [0.001],  # Minimum improvement to reset patience

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [4],
        "val_size": [0.2],
    },


    # ------ new net ----------

    # F1 val: 0.2250
    # F1 test: no data
    "RUN1_convnext_k4_roi30_augM_grid": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.85],
        "pp_strong_zoom_max": [1.10],
        "pp_strong_brightness": [0.15],
        "pp_strong_contrast": [0.15],
        "pp_strong_saturation": [0.15],
        "pp_strong_hue": [0.03],
        "pp_strong_random_erasing_p": [0.08],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [30],


        # Workers config
        "num_workers": [4],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["convnext_tiny"],
        "img_size": [384],
        "batch_size": [16],
        "lr": [3e-4],
        "weight_decay": [1e-3],
        "epochs": [50],
        "use_scheduler": [True],
        "use_amp": [True],

            
        # Extra Regularization (Non presenti nello stile originale, ma necessari qui)
        "dropout_rate": [0.3],
        "label_smoothing": [0.05],

        # Early Stopping
        "early_stopping": [True],   # Enable early stopping
        "early_stopping_patience": [8],  # Stop after N epochs without improvement
        "early_stopping_min_delta": [0.001],  # Minimum improvement to reset patience

        # ===== Validation
        "cv_type": ["kfold"],
        "n_splits": [4],
        "val_size": [0.2],
    },


    # F1 validaion: 0.4111
    # F1 test: no data
    "RUN2_convnext_k4_roi30_augS_grid": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [25],
        "pp_strong_zoom_min": [0.80],
        "pp_strong_zoom_max": [1.20],
        "pp_strong_brightness": [0.20],
        "pp_strong_contrast": [0.20],
        "pp_strong_saturation": [0.20],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.10],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [30],


        # Workers config
        "num_workers": [4],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["convnext_tiny"],
        "img_size": [384],
        "batch_size": [16],
        "lr": [3e-4],
        "weight_decay": [1e-3],
        "epochs": [50],
        "use_scheduler": [True],
        "use_amp": [True],

            
        # Extra Regularization (Non presenti nello stile originale, ma necessari qui)
        "dropout_rate": [0.3],
        "label_smoothing": [0.05],

        # Early Stopping
        "early_stopping": [True],   # Enable early stopping
        "early_stopping_patience": [8],  # Stop after N epochs without improvement
        "early_stopping_min_delta": [0.001],  # Minimum improvement to reset patience

        # ===== Validation
        "cv_type": ["kfold"],
        "n_splits": [4],
        "val_size": [0.2],
    },


    # F1 k1: 0.4116 (lr: 1e-4)
    # F1 k2: 0.3907
    # F1 test: 0.3702
    "RUN3_effb3_k4_roi30_augM_grid": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.85],
        "pp_strong_zoom_max": [1.10],
        "pp_strong_brightness": [0.15],
        "pp_strong_contrast": [0.15],
        "pp_strong_saturation": [0.15],
        "pp_strong_hue": [0.03],
        "pp_strong_random_erasing_p": [0.08],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [30],


        # Workers config
        "num_workers": [4],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["efficientnet_b3"],
        "img_size": [384],
        "batch_size": [16],
        "lr": [1e-4, 2e-4],
        "weight_decay": [1e-3],
        "epochs": [50],
        "use_scheduler": [True],
        "use_amp": [True],

            
        # Extra Regularization (Non presenti nello stile originale, ma necessari qui)
        "dropout_rate": [0.3],
        "label_smoothing": [0.05],

        # Early Stopping
        "early_stopping": [True],   # Enable early stopping
        "early_stopping_patience": [8],  # Stop after N epochs without improvement
        "early_stopping_min_delta": [0.001],  # Minimum improvement to reset patience

        # ===== Validation
        "cv_type": ["kfold"],
        "n_splits": [4],
        "val_size": [0.2],
    },


    # F1 k1: 0.3907
    # F1 k2: 0.4029 (wd: 1e-3)
    # F1 test: 0.35
    "RUN4_resnet50_k4_roi30_anchor_grid": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.85],
        "pp_strong_zoom_max": [1.10],
        "pp_strong_brightness": [0.15],
        "pp_strong_contrast": [0.15],
        "pp_strong_saturation": [0.15],
        "pp_strong_hue": [0.03],
        "pp_strong_random_erasing_p": [0.08],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [30],


        # Workers config
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [32],
        "lr": [1e-4],
        "weight_decay": [5e-4, 1e-3],
        "epochs": [50],
        "use_scheduler": [True],
        "use_amp": [True],

            
        # Extra Regularization (Non presenti nello stile originale, ma necessari qui)
        # Nota: dropout_rate non c'era nel RUN4 originale, l'ho lasciato assente per rispetto dei parametri originali.
        "label_smoothing": [0.05],

        # Early Stopping
        "early_stopping": [True],   # Enable early stopping
        "early_stopping_patience": [8],  # Stop after N epochs without improvement
        "early_stopping_min_delta": [0.001],  # Minimum improvement to reset patience

        # ===== Validation
        "cv_type": ["kfold"],
        "n_splits": [4],
        "val_size": [0.2],
    },

    #Mean F1 over 4 folds: 0.3871
    # F1 testing: 0.3567
    "RUN1B_convnext_k4_roi30_augSoft_rescue": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        "use_masks": [True],
        "mask_mode": ["crop_bbox"],

        # ===== Preprocessing config
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],

        "pp_augmentation_enabled": [True],

        # ===== OFFLINE AUG: più “soft” (meno distruttiva)
        "pp_num_aug_copies": [1],
        "pp_strong_rotation_degrees": [10],
        "pp_strong_zoom_min": [0.90],
        "pp_strong_zoom_max": [1.10],
        "pp_strong_brightness": [0.10],
        "pp_strong_contrast": [0.10],
        "pp_strong_saturation": [0.10],
        "pp_strong_hue": [0.02],
        "pp_strong_random_erasing_p": [0.03],   # << abbassato tanto

        # Smart discard / tiles
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],

        # ROI
        "use_roi_crop": [True],
        "roi_padding": [30],

        # Workers
        "num_workers": [4],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],

        "execute": [True],

        # ===== Model / training
        "backbone": ["convnext_tiny"],
        "img_size": [384],
        "batch_size": [16],

        # LR più prudente (ConvNeXt spesso gradisce finetune più “calmo” su task piccoli)
        "lr": [1e-4],
        "weight_decay": [5e-4],

        "epochs": [50],
        "use_scheduler": [True],
        "use_amp": [True],

        # ===== Regularization (ridotta)
        "dropout_rate": [0.10],        # << da 0.30 a 0.10
        "label_smoothing": [0.00],     # << tolta per vedere se stava ammazzando il segnale

        # Early stopping
        "early_stopping": [True],
        "early_stopping_patience": [8],
        "early_stopping_min_delta": [0.001],

        # Validation
        "cv_type": ["kfold"],
        "n_splits": [4],
        "val_size": [0.2],
    },


    "RUN5_convnext_k4_roi30_augMid_nogrid": {
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        "use_masks": [True],
        "mask_mode": ["crop_bbox"],

        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],

        "pp_augmentation_enabled": [True],
        "pp_num_aug_copies": [1],

        # Mid between RUN1 (too weak) and RUN2 (strong)
        "pp_strong_rotation_degrees": [20],
        "pp_strong_zoom_min": [0.85],
        "pp_strong_zoom_max": [1.15],
        "pp_strong_brightness": [0.18],
        "pp_strong_contrast": [0.18],
        "pp_strong_saturation": [0.18],
        "pp_strong_hue": [0.04],
        "pp_strong_random_erasing_p": [0.06],  # reduced vs 0.10

        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],

        "use_roi_crop": [True],
        "roi_padding": [30],

        "num_workers": [4],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],

        "execute": [True],

        "backbone": ["convnext_tiny"],
        "img_size": [384],
        "batch_size": [16],

        # Keep the "fast" LR that worked in RUN2, but lighten regularization
        "lr": [3e-4],
        "weight_decay": [5e-4],          # less than 1e-3
        "epochs": [50],
        "use_scheduler": [True],
        "use_amp": [True],

        "dropout_rate": [0.15],          # less than 0.30
        "label_smoothing": [0.02],       # small, not 0.05

        "early_stopping": [True],
        "early_stopping_patience": [8],
        "early_stopping_min_delta": [0.001],

        "cv_type": ["kfold"],
        "n_splits": [4],
        "val_size": [0.2],
    },


    "EFFB3_k2_roi30_augS_nogrid_push": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        "use_masks": [True],
        "mask_mode": ["crop_bbox"],

        # ===== Preprocessing config
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],

        # ===== OFFLINE AUG (enabled)
        "pp_augmentation_enabled": [True],
        "pp_num_aug_copies": [1],

        # “AugS” (moderatamente strong, non distruttiva)
        "pp_strong_rotation_degrees": [25],
        "pp_strong_zoom_min": [0.80],
        "pp_strong_zoom_max": [1.20],
        "pp_strong_brightness": [0.20],
        "pp_strong_contrast": [0.20],
        "pp_strong_saturation": [0.20],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.08],   # leggermente meno di 0.10

        # Smart discard / tiles
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [30],

        # Workers config (una sola volta!)
        "num_workers": [4],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],

        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["efficientnet_b3"],
        "img_size": [384],
        "batch_size": [16],

        # LR “safe” per EffB3 su dataset piccoli
        "lr": [1e-4],
        "weight_decay": [1e-3],
        "epochs": [50],
        "use_scheduler": [True],
        "use_amp": [True],

        # Regularization (moderata)
        "dropout_rate": [0.35],        # un filo più di 0.30
        "label_smoothing": [0.05],

        # Early stopping
        "early_stopping": [True],
        "early_stopping_patience": [8],
        "early_stopping_min_delta": [0.001],

        # ===== Validation
        "cv_type": ["kfold"],
        "n_splits": [2],
        "val_size": [0.2],  # ignorata per kfold, ok lasciarla
    },

    "best_config": {
        # ===== Dataset (uses output of preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],
        # ingnored with pp_*
        "use_masks": [True], 
        "mask_mode": ["crop_bbox"],



        # ===== Preprocessing config (one value per key)
        "pp_remove_shrek": [True],
        "pp_fix_stained": [True],
        "pp_split_doubles": [True],
        "pp_remove_black_rect": [True],
        "pp_padding_square": [False],
        "pp_crop_to_mask": [False],
        "pp_crop_padding": [10],
        "pp_target_size": [384],


        "pp_augmentation_enabled": [True],

        # =====AUGMENTATION =====
        # Strong augment params (used only if pp_augmentation_enabled=True)
        "pp_num_aug_copies": [4],
        "pp_strong_rotation_degrees": [45],
        "pp_strong_zoom_min": [1.0],
        "pp_strong_zoom_max": [1.5],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # Smart discard / masks
        "pp_smart_discard_threshold": [0.02],
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [False],  # keep False if use_roi_crop=True

        # =======================

        # ROI strategy
        "use_roi_crop": [True],
        "roi_padding": [10],


        # Workers config
        "num_workers": [2],
        "persistent_workers": [True],
        "pin_memory": [True],
        "prefetch_factor": [2],


        # Execute preprocessing
        "execute": [True],

        # ===== Model / training hyperparams
        "backbone": ["resnet50"],
        "img_size": [384],
        "batch_size": [16],
        "lr": [1e-4],
        "weight_decay": [1e-3],
        "epochs": [50],
        "use_scheduler": [True],
        "use_masks": [True],

            
        # Extra Regularization (Non presenti nello stile originale, ma necessari qui)
        "dropout_rate": [0.4],
        "label_smoothing": [0.1],

        # Early Stopping
        "early_stopping": [True],   # Enable early stopping
        "early_stopping_patience": [5],  # Stop after N epochs without improvement
        "early_stopping_min_delta": [0.001],  # Minimum improvement to reset patience

        # ===== Validation
        "cv_type": ["kfold"],
        "n_splits": [4],
        "val_size": [0.2],
    },


}




