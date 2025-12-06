# includes/config.py

from dataclasses import dataclass
from typing import Optional, Dict


@dataclass
class TrainingConfig:
    """Configuration object for the Doctogres challenge."""

    # --- Experiment name (used also for submission filename) ---
    exp_name: str = "baseline"

    # --- Project paths (environment-agnostic) ---
    # project_root: base directory of the project (set in notebook)
    project_root: str = "."
    # data_dir: folder that contains train_data, test_data, train_labels.csv
    data_dir: str = "data"
    # out_dir: folder where to save submissions, logs, etc.
    out_dir: str = "out"

    # inside data_dir
    train_img_dir: str = "train_data"
    test_img_dir: str = "test_data"
    labels_csv: str = "train_labels.csv"

    # masks folder (inside data_dir). If None -> same as train_img_dir
    mask_dir: Optional[str] = None

    # --- Model / data ---
    # backbone: "resnet18", "resnet50", "efficientnet_b0"
    backbone: str = "resnet18"
    img_size: int = 224
    batch_size: int = 16
    num_workers: int = 4
    val_size: float = 0.2
    random_seed: int = 42

    # How to use masks when use_masks=True:
    #   "crop_bbox"  -> crop using bounding box of mask
    #   "multiply"   -> multiply image by mask (img * mask)
    mask_mode: str = "crop_bbox"

    # --- Training ---
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    use_scheduler: bool = True
    use_masks: bool = False          # True = use masks according to mask_mode


# ---------- Predefined experiment configs ----------

# Baseline: ResNet18, 224x224, no masks
BASELINE = TrainingConfig(
    exp_name="baseline",
    backbone="resnet18",
    img_size=224,
    use_masks=False,
    mask_mode="crop_bbox",
)

# ResNet50, bigger images, more epochs, no masks
RESNET50_BIG = TrainingConfig(
    exp_name="resnet50_big",
    backbone="resnet50",
    img_size=384,
    batch_size=16,
    epochs=50,
    use_masks=False,
    mask_mode="crop_bbox",
)

# ResNet50, bigger images, masks ON (crop bbox)
RESNET50_BIG_MASKS = TrainingConfig(
    exp_name="resnet50_big_masks",
    backbone="resnet50",
    img_size=384,
    batch_size=16,
    epochs=50,
    use_masks=True,
    mask_mode="crop_bbox",
)

# ---------- Challenge 2-2 style experiments ----------

# prima si chiamava CHALLENGE_2_2
EFFB0_224_MASKMUL_F1 = TrainingConfig(
    exp_name="effb0_224_maskmul_f1",
    backbone="efficientnet_b0",
    img_size=224,
    batch_size=16,
    epochs=50,
    use_masks=True,
    mask_mode="multiply",
)

EXPERIMENTS: Dict[str, TrainingConfig] = {
    "baseline": BASELINE,
    "resnet50_big": RESNET50_BIG,
    "resnet50_big_masks": RESNET50_BIG_MASKS,
    "effb0_224_maskmul_f1": EFFB0_224_MASKMUL_F1,
}
