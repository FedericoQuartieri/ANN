# includes/model_utils.py

from typing import Dict, Any, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import f1_score  # F1 macro

from .config import TrainingConfig


def build_model(cfg: TrainingConfig, num_classes: int, device: torch.device) -> nn.Module:
    """Create a classification model given the backbone name."""
    backbone = cfg.backbone.lower()

    if backbone == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif backbone == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    elif backbone == "efficientnet_b0":
        # EfficientNet-B0 from torchvision
        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        # classifier is usually: Dropout -> Linear
        in_features = model.classifier[1].in_features
        dropout_rate = getattr(cfg, 'dropout_rate', 0.0)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    elif backbone == "efficientnet_b3":
        # EfficientNet-B3 from torchvision (better for 384x384 images)
        model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        # classifier is usually: Dropout -> Linear
        in_features = model.classifier[1].in_features
        dropout_rate = getattr(cfg, 'dropout_rate', 0.0)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    elif backbone == "convnext_tiny":
        # ConvNeXt-Tiny from torchvision (modern CNN, good accuracy/speed tradeoff)
        model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        )
        # classifier is: LayerNorm -> Flatten -> Linear
        in_features = model.classifier[2].in_features
        dropout_rate = getattr(cfg, 'dropout_rate', 0.0)
        model.classifier[2] = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes)
        )

    else:
        raise ValueError(f"Unknown backbone: {cfg.backbone}")

    model = model.to(device)
    return model


def create_criterion_optimizer_scheduler(
    cfg: TrainingConfig,
    model: nn.Module,
    train_df,
    device: torch.device,
):
    """Create loss function, optimizer and (optional) scheduler."""
    # Compute class weights from training set
    class_counts = train_df["label_idx"].value_counts().sort_index().values.astype(float)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)

    class_weights_tensor = torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )

    # Label smoothing for better generalization (0.0 = disabled)
    label_smoothing = getattr(cfg, 'label_smoothing', 0.0)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=label_smoothing
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = None
    if cfg.use_scheduler:
        # adesso scheduler segue la F1 (val_f1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=2,
        )

    return criterion, optimizer, scheduler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int | None = None,
    num_epochs: int | None = None,
    use_amp: bool = True,
) -> Tuple[float, float]:
    """Train model for one epoch with optional mixed-precision training.

    Args:
        use_amp: Enable automatic mixed precision (float16). Recommended for modern GPUs.

    Ritorna:
        epoch_loss, epoch_f1_macro
    """
    model.train()
    running_loss = 0.0
    running_total = 0

    all_labels = []
    all_preds = []

    # Mixed-precision training setup
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    # Descrizione per la barra
    if epoch is not None and num_epochs is not None:
        desc = f"Epoch {epoch}/{num_epochs}"
    else:
        desc = "Train"

    # Proviamo a usare tqdm per avere una barra grafica
    use_tqdm = False
    try:
        from tqdm.auto import tqdm  # type: ignore

        iterator = tqdm(loader, desc=desc, leave=False)
        use_tqdm = True
    except Exception:
        iterator = loader

    n_batches = len(loader)

    for batch_idx, batch in enumerate(iterator, start=1):
        # Support both (img, label) and (img, label, case_id) formats
        if len(batch) == 3:
            images, labels, _ = batch  # Ignore case_id during training
        else:
            images, labels = batch
        
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Mixed-precision forward pass
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # Scaled backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Compute predictions AFTER autocast (use float32 for stability)
            with torch.no_grad():
                _, preds = torch.max(outputs.float(), dim=1)
        else:
            # Standard float32 training
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # Compute predictions
            with torch.no_grad():
                _, preds = torch.max(outputs, dim=1)

        # accumulo per loss
        running_loss += loss.item() * images.size(0)
        running_total += labels.size(0)

        # accumulo per F1
        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

        # metriche parziali
        y_true_partial = np.concatenate(all_labels)
        y_pred_partial = np.concatenate(all_preds)
        partial_f1 = f1_score(y_true_partial, y_pred_partial, average="macro")
        avg_loss = running_loss / max(running_total, 1)

        if use_tqdm:
            # tqdm: aggiorno postfix
            iterator.set_postfix(
                loss=f"{avg_loss:.4f}",
                f1=f"{partial_f1:.4f}",
            )
        else:
            # Fallback senza tqdm: stampa ogni 10 batch (o inizio/fine)
            if batch_idx == 1 or batch_idx == n_batches or batch_idx % 10 == 0:
                print(
                    f"    [Batch {batch_idx:3d}/{n_batches}] "
                    f"loss={avg_loss:.4f}  f1={partial_f1:.4f}",
                    end="\r",
                    flush=True,
                )

    if not use_tqdm:
        print()  # newline dopo la barra "manuale"

    epoch_loss = running_loss / running_total
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    epoch_f1 = f1_score(y_true, y_pred, average="macro")

    return epoch_loss, epoch_f1


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    aggregate_by_image: bool = True,
):
    """Evaluate model on a loader, returning metrics and predictions.
    
    If aggregate_by_image=True, tiles are aggregated by original image
    (using mean softmax probabilities) before computing F1. This makes
    validation metrics comparable to test metrics.

    Ritorna:
        epoch_loss, epoch_f1_macro, y_true, y_pred
    """
    model.eval()
    running_loss = 0.0
    running_total = 0

    all_labels = []
    all_preds = []
    all_probs = []  # For image-level aggregation
    all_case_ids = []  # Original image IDs

    with torch.no_grad():
        for batch in loader:
            # Support both (img, label) and (img, label, case_id) formats
            if len(batch) == 3:
                images, labels, case_ids = batch
            else:
                images, labels = batch
                case_ids = [f"sample_{i}" for i in range(len(labels))]  # Fallback
            
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)

            running_loss += loss.item() * images.size(0)
            running_total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_case_ids.extend(case_ids)

    epoch_loss = running_loss / running_total
    
    if aggregate_by_image and len(set(all_case_ids)) < len(all_case_ids):
        # There are multiple tiles per image -> aggregate
        # Group by case_id
        from collections import defaultdict
        grouped_probs = defaultdict(list)
        grouped_labels = {}
        
        for case_id, prob, label in zip(all_case_ids, all_probs, all_labels):
            grouped_probs[case_id].append(prob)
            grouped_labels[case_id] = label  # All tiles of same image have same label
        
        # Aggregate: mean of probabilities per image, then argmax
        y_true = []
        y_pred = []
        for case_id in sorted(grouped_probs.keys()):
            probs_arr = np.array(grouped_probs[case_id])
            avg_prob = np.mean(probs_arr, axis=0)
            pred = int(np.argmax(avg_prob))
            y_pred.append(pred)
            y_true.append(grouped_labels[case_id])
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    else:
        # No aggregation (single tile per image or disabled)
        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
    
    epoch_f1 = f1_score(y_true, y_pred, average="macro")

    return epoch_loss, epoch_f1, y_true, y_pred


def train_model(
    cfg: TrainingConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> Tuple[Dict[str, Any], Dict[str, list]]:
    """Main training loop: returns best_state_dict and a history dict.

    Usa la F1 macro come metrica principale.
    """

    exp_name = getattr(cfg, "exp_name", "experiment")
    print("==============================================================")
    print(f"Starting training - experiment: {exp_name}")
    print(f"Backbone: {cfg.backbone}  |  img_size: {cfg.img_size}  |  epochs: {cfg.epochs}")
    print("==============================================================")

    best_val_f1 = 0.0
    best_state_dict = None
    
    # Early stopping setup
    early_stopping = getattr(cfg, 'early_stopping', False)
    es_patience = getattr(cfg, 'early_stopping_patience', 5)
    es_min_delta = getattr(cfg, 'early_stopping_min_delta', 0.001)
    es_counter = 0  # Epochs without improvement
    es_best_f1 = 0.0
    
    if early_stopping:
        print(f"✓ Early stopping enabled (patience={es_patience}, min_delta={es_min_delta})")

    history = {
        "train_loss": [],
        "train_f1": [],
        "val_loss": [],
        "val_f1": [],
        # per compatibilità, teniamo anche questi nomi
        "train_acc": [],
        "val_acc": [],
    }

    # Mixed precision from config (default: auto-enable on CUDA)
    use_amp = getattr(cfg, 'use_amp', device.type == 'cuda')
    if use_amp and device.type == 'cuda':
        print("✓ Mixed-precision training (AMP) enabled - using float16 for faster training")
    elif use_amp and device.type != 'cuda':
        print("⚠ use_amp=True but device is CPU - disabling AMP")
        use_amp = False
    
    for epoch in range(1, cfg.epochs + 1):
        print("\n--------------------------------------------------------------")
        print(f"[Epoch {epoch}/{cfg.epochs}]")

        # barra di progresso per il training
        train_loss, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            num_epochs=cfg.epochs,
            use_amp=use_amp,
        )
        val_loss, val_f1, val_true, val_pred = evaluate(
            model, val_loader, criterion, device
        )

        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        # alias
        history["train_acc"].append(train_f1)
        history["val_acc"].append(val_f1)

        # Stile compatto: F1 al posto di accuracy
        print(f"  Train - loss: {train_loss:.4f}  |  f1: {train_f1:.4f}")
        print(f"  Val   - loss: {val_loss:.4f}  |  f1: {val_f1:.4f}")

        if scheduler is not None:
            scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state_dict = model.state_dict().copy()
            print(f"  >> New best model! val_f1 improved to {best_val_f1:.4f}")
        
        # Early stopping check
        if early_stopping:
            if val_f1 > es_best_f1 + es_min_delta:
                es_best_f1 = val_f1
                es_counter = 0
            else:
                es_counter += 1
                print(f"  >> Early stopping: {es_counter}/{es_patience} (no improvement)")
                
            if es_counter >= es_patience:
                print(f"\n⚠ Early stopping triggered at epoch {epoch}!")
                print(f"  No improvement for {es_patience} consecutive epochs.")
                break

    print("\n==============================================================")
    print(f"Training finished for experiment: {exp_name}")
    print(f"Best validation F1 (macro): {best_val_f1:.4f}")
    if early_stopping and es_counter >= es_patience:
        print(f"Stopped early at epoch {epoch}/{cfg.epochs}")
    print("==============================================================")

    return best_state_dict, history
