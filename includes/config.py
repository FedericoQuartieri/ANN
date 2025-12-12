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
    random_seed: int = 42


# cv_type: ["holdout", "kfold"]
# augmentation: ["none", "strong"]
# mask_mode: ["multiply", "crop_bbox"]

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
        "pp_darken_outside_mask": [True],
        "pp_augmentation_enabled": [True],

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
        "pp_split_into_tiles": [True],
        "pp_remove_empty_masks": [True],
        "pp_darken_outside_mask": [True],


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
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },


    # ----------------------------------------------------------------------
    # ResNet50 su dataset ORIGINALE, SENZA preprocessing offline
    # (usa direttamente train_data / test_data / train_labels.csv)
    # ----------------------------------------------------------------------
    
    # 0.1865
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


    # 0.2825
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
        "pp_darken_outside_mask": [True],


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
        "pp_darken_outside_mask": [True],


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
        "pp_darken_outside_mask": [False],
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
        "pp_darken_outside_mask": [True],


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


    # k1 F1: 0.4037
    # 
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
        "pp_darken_outside_mask": [True],


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




    "resnet50_424_augmentation_weight-epoch-grid": {
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
        "pp_augmentation_enabled": [True],

        "pp_crop_padding": [10],
        "pp_target_size": [424],
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
        "pp_darken_outside_mask": [True],


        "execute" : [True],
        # ===== Hyperparam
        "backbone": ["resnet50"],
        "img_size": [424],
        "batch_size": [16],
        "num_workers": [4],
        "lr": [1e-4],
        "weight_decay": [1e-4, 1e-3, 5e-3],        
        "epochs": [40, 60],
        "use_scheduler": [True],
        "use_amp": [True],

        # ===== Validation
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },


    "resnet50_424_lightpreproc_nooffaug_weight-epoch-grid": {
        # ===== Dataset (sempre lo stesso output di preprocessing.py)
        "train_img_dir": ["pp_train_data"],
        "test_img_dir": ["pp_test_data"],
        "labels_csv": ["pp_train_labels.csv"],

        # mask a runtime ignorate con pp_*
        "mask_mode": ["crop_bbox"],  # ignored
        "use_masks": [False],        # esplicito che non le usiamo

        # ==== PREPROCESSING CONFIG - DRASTICAMENTE DIVERSA ====
        # -> molto meno “surgical”, niente tiles, niente darken, niente strong aug

        "pp_remove_shrek": [True],       # Shrek lo togliamo comunque
        "pp_fix_stained": [False],       # NON correggiamo il verde (distribuzione colori diversa)
        "pp_split_doubles": [False],     # NON splittiamo le doppie: immagini più grandi / contesto
        "pp_remove_black_rect": [False], # NON rimuoviamo rettangoli neri (meno crop aggressivo)

        # padding a quadrato + niente crop_to_mask
        "pp_padding_square": [True],
        "pp_crop_to_mask": [False],

        # == AUGMENTATION OFFLINE DISABILITATA ==
        "pp_augmentation_enabled": [False],

        # comunque servono per resize / shape
        "pp_crop_padding": [0],          # nessun padding extra sul bbox (non usato se crop_to_mask=False)
        "pp_target_size": [424],

        # parametri di aug restano ma non verranno usati
        "pp_strong_rotation_degrees": [15],
        "pp_strong_zoom_min": [0.9],
        "pp_strong_zoom_max": [1.1],
        "pp_strong_brightness": [0.2],
        "pp_strong_contrast": [0.2],
        "pp_strong_saturation": [0.2],
        "pp_strong_hue": [0.05],
        "pp_strong_random_erasing_p": [0.1],

        # niente smart discard, niente tiles, niente darken
        "pp_smart_discard_threshold": [0.0],  # 0 => non scarta nulla
        "pp_split_into_tiles": [False],
        "pp_remove_empty_masks": [False],
        "pp_darken_outside_mask": [False],

        "execute": [True],

        # ===== Hyperparam (IDENTICI all’altra config) =====
        "backbone": ["resnet50"],
        "img_size": [424],
        "batch_size": [16],
        "num_workers": [4],
        "lr": [1e-4],
        "weight_decay": [1e-4, 1e-3, 5e-3],
        "epochs": [40, 60],
        "use_scheduler": [True],
        "use_amp": [True],

        # ===== Validation =====
        "cv_type": ["holdout"],
        "n_splits": [5],
        "val_size": [0.2],
    },


}



        

## AGGIUNGERE DROPOUT ?????