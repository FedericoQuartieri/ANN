# includes/inference_utils.py

import os
from typing import Dict, List, Tuple

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
):
    """Run inference on the test set and save a submission CSV.

    - If output_csv is None, use submission_{exp_name}.csv
    - Se siamo su Colab, prova anche a fare il download del file.
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

    submission_df = pd.DataFrame(
        {"sample_index": all_names, "label": labels}
    ).sort_values("sample_index")

    # ---------- nome file in base all'esperimento ----------
    if output_csv is None:
        exp_name = getattr(cfg, "exp_name", "experiment")
        safe_name = str(exp_name).replace(" ", "_")
        output_csv = f"submission_{safe_name}.csv"

    save_dir = _get_out_root(cfg)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, output_csv)
    submission_df.to_csv(out_path, index=False)
    print("Saved submission to:", out_path)

    # ---------- download automatico se siamo su Colab ----------
    try:
        from google.colab import files  # type: ignore
        files.download(out_path)
        print("Triggered Colab download for:", out_path)
    except Exception:
        # non siamo su Colab / nessun download possibile
        pass
