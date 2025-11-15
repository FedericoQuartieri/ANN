# pirate_pain_baseline.py
# Train-from-scratch baseline for Pirate Pain (multivariate time-series classification)
# Requires: pandas, numpy, scikit-learn, torch, tqdm
# Tested on CPU/MPS (Apple Silicon) and CUDA if available.

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ---------------------------
# Utils
# ---------------------------
def seed_everything(seed=42):
    import random, os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ---------------------------
# Data shaping
# ---------------------------
def infer_columns(df):
    id_col = "sample_index"
    time_col = "time" if "time" in df.columns else None
    static_candidates = ["n_legs","n_hands","n_eyes"]
    static_cols = [c for c in static_candidates if c in df.columns]
    ignore = set([id_col] + ([time_col] if time_col else []) + static_cols)
    feature_cols = [c for c in df.columns if c not in ignore]
    return id_col, time_col, static_cols, feature_cols

def _numericize_features(df, cols):
    """Return a numeric version of df[cols], mapping common words to numbers and dropping all-NaN cols."""
    mapping = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "true": 1, "false": 0, "yes": 1, "no": 0,
        "none": None, "null": None, "nan": None, "": None
    }
    out = df[cols].copy()
    for c in out.columns:
        if out[c].dtype == object:
            s = out[c].astype(str).str.strip().str.lower().replace(mapping)
            out[c] = pd.to_numeric(s, errors="coerce")
        else:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    keep = out.columns[out.notna().any()].tolist()
    dropped = [c for c in out.columns if c not in keep]
    if dropped:
        print(f"[build_sequences] Dropping non-numeric/all-NaN features: {dropped[:15]}" + (" ..." if len(dropped) > 15 else ""))
    return out[keep], keep

def build_sequences(X_df, y_df=None, expect_T=180):
    id_col = "sample_index"
    time_col = "time" if "time" in X_df.columns else None
    static_candidates = ["n_legs","n_hands","n_eyes"]
    static_cols = [c for c in static_candidates if c in X_df.columns]
    ignore = set([id_col] + ([time_col] if time_col else []) + static_cols)
    ignore |= {"label","target","class"}
    raw_dyn_cols = [c for c in X_df.columns if c not in ignore]

    if time_col is not None:
        X_df = X_df.sort_values([id_col, time_col])
    else:
        X_df = X_df.sort_values([id_col])

    dyn_numeric, dyn_cols = _numericize_features(X_df, raw_dyn_cols)
    for c in dyn_cols:
        X_df[c] = dyn_numeric[c].values

    groups = X_df.groupby(id_col)
    sample_ids = []
    lengths = []                     # <--- NEW
    X_dyn_list, X_static_list = [], []

    def fix_len(arr, T):
        n = arr.shape[0]
        if n == T: return arr, "ok"
        if n > T:  return arr[-T:, :], "trunc"
        pad = np.repeat(arr[-1:, :], T - n, axis=0)
        return np.concatenate([arr, pad], axis=0), "pad"

    n_ok = n_pad = n_trunc = 0

    for s_id, g in groups:
        g_dyn = g[dyn_cols].ffill().bfill().fillna(0.0)
        arr0 = g_dyn.to_numpy(dtype=np.float32)
        true_len = arr0.shape[0]           # <--- NEW (pre padding)
        arr, tag = fix_len(arr0, expect_T)
        if tag == "ok": n_ok += 1
        elif tag == "pad": n_pad += 1
        else: n_trunc += 1

        if len(static_cols) > 0:
            s0 = g[static_cols].iloc[0]
            s0 = s0.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            s = s0.to_numpy(dtype=np.float32)
        else:
            s = np.zeros(0, dtype=np.float32)

        sample_ids.append(s_id)
        lengths.append(min(true_len, expect_T))   # cap to T
        X_dyn_list.append(arr)
        X_static_list.append(s)

    if len(X_dyn_list) == 0:
        raise ValueError("No sequences assembled. Check 'sample_index' and that each sample has rows.")

    X_dyn = np.stack(X_dyn_list, axis=0)
    X_static = np.stack(X_static_list, axis=0)
    sample_ids = np.array(sample_ids)
    lengths = np.array(lengths, dtype=np.int64)   # <--- NEW

    print(f"[build_sequences] Target T={expect_T} -> ok:{n_ok}  padded:{n_pad}  truncated:{n_trunc}")

    y = None
    classes = None
    if y_df is not None:
        label_cols = [c for c in y_df.columns if c != "sample_index"]
        assert len(label_cols) == 1, "y_train must have one target column besides sample_index"
        target_col = label_cols[0]
        y_map = y_df.set_index("sample_index")[target_col].to_dict()
        y_raw = [y_map[s] for s in sample_ids]
        classes = sorted(list(set(y_raw)))
        class_to_idx = {c:i for i,c in enumerate(classes)}
        y = np.array([class_to_idx[v] for v in y_raw], dtype=np.int64)

    return X_dyn, X_static, sample_ids, lengths, y, classes, dyn_cols, static_cols

class SequenceDataset(Dataset):
    def __init__(self, X_dyn, X_static, lengths, y=None, train=False, aug_p=0.0):
        self.X_dyn = X_dyn
        self.X_static = X_static
        self.lengths = lengths
        self.y = y
        self.train = train
        self.aug_p = aug_p

    def _augment(self, x):  # x: [T,C]
        # light, safe defaults
        import numpy as np
        T, C = x.shape
        if np.random.rand() < self.aug_p:
            x = x + np.random.normal(0, 0.01, size=x.shape)        # jitter
        if np.random.rand() < self.aug_p:
            scale = 1.0 + np.random.normal(0, 0.05, size=(1, C))   # channel scaling
            x = x * scale
        if np.random.rand() < self.aug_p:
            w = np.random.randint(5, 20)
            s = np.random.randint(0, T - w)
            x[s:s+w, :] = 0                                        # time mask
        return x

    def __len__(self):
        return self.X_dyn.shape[0]

    def __getitem__(self, idx):
        x_dyn = self.X_dyn[idx]      # np array [T,C]
        if self.train and self.aug_p > 0:
            x_dyn = self._augment(x_dyn.copy()).astype(np.float32)
        x_dyn = torch.from_numpy(x_dyn)
        x_static = torch.from_numpy(self.X_static[idx])
        length = int(self.lengths[idx])
        if self.y is None:
            return x_dyn, x_static, length
        return x_dyn, x_static, length, int(self.y[idx])

class AttnPool(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d//2), nn.Tanh(), nn.Linear(d//2, 1))
    def forward(self, x, mask):                 # x:[B,T,D], mask:[B,T] bool
        a = self.proj(x).squeeze(-1)            # [B,T]
        a = a.masked_fill(~mask, float('-inf'))
        w = a.softmax(dim=1)                    # [B,T]
        return (x * w.unsqueeze(-1)).sum(1)     # [B,D]

# ---------------------------
# Model (CNN + BiGRU head)
# ---------------------------
class PirateNet(nn.Module):
    def __init__(self, c_dyn, c_static, hidden=64, rnn_layers=1, num_classes=3, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(c_dyn, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(input_size=64, hidden_size=hidden, num_layers=rnn_layers,
                          batch_first=True, bidirectional=True)
        self.attn = AttnPool(2*hidden)

        static_out = 32 if c_static > 0 else 0
        if c_static > 0:
            self.static_mlp = nn.Sequential(
                nn.LayerNorm(c_static),
                nn.Linear(c_static, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, static_out),
                nn.ReLU(),
            )

        head_in = (2*hidden)*3 + static_out  # mean + max + attn
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(head_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_dyn, x_static, lengths):
        # x_dyn: [B,T,C] -> conv -> [B,T,64] -> BiGRU -> [B,T,2H]
        x = self.conv(x_dyn.transpose(1,2)).transpose(1,2)
        out, _ = self.rnn(x)

        B, T, D = out.shape
        device = out.device
        # mask: True on real steps, False on padding
        lens = lengths.to(device)
        ar = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        mask = ar < lens.unsqueeze(1)                         # [B,T] bool

        # masked pools
        h_mean = (out * mask.unsqueeze(-1)).sum(1) / torch.clamp(lens.unsqueeze(1), min=1).to(out.dtype)
        out_masked = out.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        h_max = out_masked.max(1).values
        h_attn = self.attn(out, mask)

        feat = torch.cat([h_mean, h_max, h_attn], dim=1)
        if x_static is not None and x_static.shape[1] > 0:
            s = self.static_mlp(x_static)
            feat = torch.cat([feat, s], dim=1)
        return self.head(feat)

# ---------------------------
# Training / Evaluation
# ---------------------------
def train_one_epoch(model, loader, optimizer, device, criterion, scheduler=None):
    model.train()
    total_loss = 0.0
    preds, trues = [], []
    for batch in loader:
        xb_dyn, xb_static, xlens, yb = batch
        xb_dyn = xb_dyn.to(device)
        xb_static = xb_static.to(device)
        xlens = torch.as_tensor(xlens, device=device)
        yb = torch.as_tensor(yb, device=device)

        optimizer.zero_grad()
        logits = model(xb_dyn, xb_static, xlens)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item() * yb.size(0)
        preds.append(logits.detach().softmax(dim=1).cpu().numpy())
        trues.append(yb.detach().cpu().numpy())

    preds = np.concatenate(preds); trues = np.concatenate(trues)
    f1 = f1_score(trues, preds.argmax(1), average="macro")
    acc = accuracy_score(trues, preds.argmax(1))
    return total_loss / len(loader.dataset), f1, acc

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    for batch in loader:
        if len(batch) == 3:     # test loader
            xb_dyn, xb_static, xlens = batch
            yb = None
        else:
            xb_dyn, xb_static, xlens, yb = batch
            yb = torch.as_tensor(yb, device=device)
        xb_dyn = xb_dyn.to(device)
        xb_static = xb_static.to(device)
        xlens = torch.as_tensor(xlens, device=device)

        logits = model(xb_dyn, xb_static, xlens)
        if yb is not None:
            loss = criterion(logits, yb)
            total_loss += loss.item() * yb.size(0)
            trues.append(yb.detach().cpu().numpy())
        preds.append(logits.detach().softmax(dim=1).cpu().numpy())

    preds = np.concatenate(preds)
    if trues:
        trues = np.concatenate(trues)
        f1 = f1_score(trues, preds.argmax(1), average="macro")
        acc = accuracy_score(trues, preds.argmax(1))
        return total_loss / len(loader.dataset), f1, acc, preds
    return None, None, None, preds


def add_deltas(X):  # X: [N,T,C]
    d1 = np.diff(X, axis=1, prepend=X[:, :1, :])
    d2 = np.diff(d1, axis=1, prepend=d1[:, :1, :])
    return np.concatenate([X, d1, d2], axis=2)

# ---------------------------
# Main
# ---------------------------
def main(args):
    seed_everything(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Load data
    X_train = pd.read_csv("data/pirate_pain_train.csv")
    y_train = pd.read_csv("data/pirate_pain_train_labels.csv")
    X_test  = pd.read_csv("data/pirate_pain_test.csv")

    # Build sequences
    Xdyn_tr, Xsta_tr, ids_tr, len_tr, y, classes, dyn_cols, static_cols = build_sequences(X_train, y_train, expect_T=180)
    Xdyn_te, Xsta_te, ids_te, len_te, _, _, _, _ = build_sequences(X_test, None, expect_T=180)
    num_classes = len(classes)
    print(f"Train sequences: {len(ids_tr)}  Test sequences: {len(ids_te)}")
    print(f"Dynamic channels: {Xdyn_tr.shape[-1]}  Static dims: {Xsta_tr.shape[-1]}  Classes: {classes}")

    Xdyn_tr = add_deltas(Xdyn_tr)
    Xdyn_te = add_deltas(Xdyn_te)

    # CV setup
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # OOF storage
    oof_pred = np.zeros((len(ids_tr), num_classes), dtype=np.float32)
    test_pred_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(ids_tr, y), start=1):
        print(f"\n========== FOLD {fold}/{args.folds} ==========")
        # Fit scalers on train fold ONLY (flatten over time for per-feature scaling)
        T, C = Xdyn_tr.shape[1], Xdyn_tr.shape[2]
        dyn_scaler = StandardScaler()
        dyn_scaler.fit(Xdyn_tr[tr_idx].reshape(-1, C))
        Xdyn_tr_scaled = dyn_scaler.transform(Xdyn_tr.reshape(-1, C)).reshape(-1, T, C)
        Xdyn_te_scaled = dyn_scaler.transform(Xdyn_te.reshape(-1, C)).reshape(-1, T, C)

        if Xsta_tr.shape[1] > 0:
            sta_scaler = StandardScaler()
            sta_scaler.fit(Xsta_tr[tr_idx])
            Xsta_tr_scaled = sta_scaler.transform(Xsta_tr)
            Xsta_te_scaled = sta_scaler.transform(Xsta_te)
        else:
            Xsta_tr_scaled = Xsta_tr
            Xsta_te_scaled = Xsta_te

        # Datasets & loaders
        ds_tr = SequenceDataset(Xdyn_tr_scaled[tr_idx], Xsta_tr_scaled[tr_idx], len_tr[tr_idx], y[tr_idx], train=True, aug_p=0.5)
        ds_va = SequenceDataset(Xdyn_tr_scaled[va_idx], Xsta_tr_scaled[va_idx], len_tr[va_idx], y[va_idx], train=False)
        dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=False)
        dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

        # Model
        model = PirateNet(c_dyn=C, c_static=Xsta_tr.shape[1], hidden=args.hidden, rnn_layers=1,
                          num_classes=num_classes, dropout=args.dropout).to(device)

        # Loss (class weights if imbalance)
        class_counts = np.bincount(y[tr_idx], minlength=num_classes) + 1  # +1 smoothing
        inv_freq = class_counts.sum() / class_counts
        # normalize to mean=1 so loss scale stays reasonable
        inv_freq = inv_freq / inv_freq.mean()
        class_weights = torch.tensor(inv_freq, dtype=torch.float32, device=device)

        #criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)  # fallback to 0 if your torch is old

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(dl_tr))

        # Training loop with early stopping on macro-F1
        best_f1, patience_left = -1.0, args.patience
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}  # init in case no improvement


        for epoch in range(1, args.epochs+1):
          tr_loss, tr_f1, tr_acc = train_one_epoch(model, dl_tr, optimizer, device, criterion, scheduler)
          va_loss, va_f1, va_acc, _ = evaluate(model, dl_va, device, criterion)

          print(f"Epoch {epoch:02d}: "
                f"train loss {tr_loss:.4f} f1 {tr_f1:.4f} acc {tr_acc:.4f} | "
                f"val loss {va_loss:.4f} f1 {va_f1:.4f} acc {va_acc:.4f}")

          if va_f1 > best_f1 + 1e-5:
              best_f1 = va_f1
              best_state = {k: v.cpu() for k, v in model.state_dict().items()}
              patience_left = args.patience
          else:
              patience_left -= 1
              if patience_left <= 0:
                  print("Early stopping.")
                  break

        # Load best
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        # Store OOF predictions
        _, _, _, va_pred = evaluate(model, dl_va, device, criterion)
        oof_pred[va_idx] = va_pred

        # Predict test for this fold
        dl_te = DataLoader(SequenceDataset(Xdyn_te_scaled, Xsta_te_scaled, len_te, None), batch_size=args.batch_size, shuffle=False)
        _, _, _, te_pred = evaluate(model, dl_te, device, criterion)
        test_pred_folds.append(te_pred)

    # Report OOF score
    oof_labels = y
    oof_f1 = f1_score(oof_labels, oof_pred.argmax(1), average="macro")
    oof_acc = accuracy_score(oof_labels, oof_pred.argmax(1))
    print(f"\nOOF macro-F1: {oof_f1:.4f} | OOF Acc: {oof_acc:.4f}")

    # Average test predictions across folds
    test_pred = np.mean(np.stack(test_pred_folds, axis=0), axis=0)  # [N_test, K]

    # Write OOF preds (optional)
    pd.DataFrame({
        "sample_index": ids_tr,
        **{f"prob_{cls}": oof_pred[:, i] for i, cls in enumerate(classes)},
        "oof_pred": [classes[i] for i in oof_pred.argmax(1)],
        "target": [classes[i] for i in oof_labels]
    }).to_csv("oof_predictions.csv", index=False)

    # Submission: match sample_submission columns if available
    submit_col_id = "sample_index"
    # Try to read sample submission for correct column names/order
    label_col_name = "label"
    if os.path.exists("sample_submission.csv"):
        sub_template = pd.read_csv("sample_submission.csv")
        submit_col_id = [c for c in sub_template.columns if c != label_col_name][0] if label_col_name in sub_template.columns else "sample_index"
        if label_col_name not in sub_template.columns:
            # try to detect
            non_id = [c for c in sub_template.columns if c != submit_col_id]
            if len(non_id) == 1:
                label_col_name = non_id[0]
    test_pred_labels = [classes[i] for i in test_pred.argmax(1)]
    submission = pd.DataFrame({submit_col_id: ids_te, label_col_name: test_pred_labels})
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args, _ = p.parse_known_args()  # ignores Jupyter/Colab's extra -f argument
    main(args)