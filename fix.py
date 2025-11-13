# notebook2py_fixer.py
# Fix converted-notebook .py files:
# 1) Replace make_loader(...) with a robust version.
# 2) Replace the get_ipython('time'...) training magic with a proper fit() call under main-guard.
# 3) Comment every line with get_ipython/get_iptyhon and any matplotlib/plt usage.
# 4) Remove any occurrence of palette=colors (handles commas & spaces).

import re
import argparse
from pathlib import Path

MAKE_LOADER_NEW = r'''
def make_loader(ds, batch_size, shuffle, drop_last, num_workers=None):
    """
    Robust DataLoader for macOS/CPU and CUDA.
    - Defaults to num_workers=0 on LOCAL/CPU to avoid hangs.
    - Enables pin_memory/prefetch only on CUDA.
    """
    import os
    import torch
    from torch.utils.data import DataLoader

    if num_workers is None:
        if 'LOCAL' in globals() and globals()['LOCAL']:
            num_workers = 0
        else:
            cpu_cores = os.cpu_count() or 2
            num_workers = min(4, max(0, cpu_cores - 1))

    kwargs = dict(
        dataset=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    if torch.cuda.is_available():
        kwargs['pin_memory'] = True
        kwargs['pin_memory_device'] = 'cuda'
        if num_workers > 0:
            kwargs['prefetch_factor'] = 4

    return DataLoader(**kwargs)
'''.lstrip()

TRAIN_SNIPPET = r'''
# === Replaced: proper training call under main-guard ===
if __name__ == "__main__":
    # Train model and track training history
    rnn_model, training_history = fit(
        model=rnn_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
        device=device,
        writer=writer,
        verbose=1,
        experiment_name="rnn",
        patience=PATIENCE,
    )

    # Update best model if current performance is superior
    if training_history and isinstance(training_history, dict) and 'val_f1' in training_history and training_history['val_f1']:
        if training_history['val_f1'][-1] > best_performance:
            best_model = rnn_model
            best_performance = training_history['val_f1'][-1]
# === End replacement ===
'''.lstrip()

def replace_make_loader(text: str) -> tuple[str, int]:
    t = text if text.endswith('\n') else text + '\n'
    pattern = re.compile(r"^def\s+make_loader\s*\([^\)]*\):[\s\S]*?(?=^\S|\Z)", re.M)
    new_text, n = re.subn(pattern, MAKE_LOADER_NEW + "\n", t)
    return new_text, n

def replace_training_magic(text: str) -> tuple[str, int]:
    pattern = re.compile(r"^.*get_ipython\(\)\.run_cell_magic\('time'.*Train model and track training history.*$", re.M)
    new_text, n = re.subn(pattern, TRAIN_SNIPPET.rstrip(), text)
    if n == 0:
        generic = re.compile(r"^.*get_ipython\(\)\.run_cell_magic\('time'.*\)$", re.M)
        new_text, n = re.subn(generic, TRAIN_SNIPPET.rstrip(), text)
    return new_text, n

def comment_selected_lines(text: str) -> tuple[str, int]:
    out_lines, count = [], 0
    for line in text.splitlines(keepends=True):
        must_comment = (
            re.search(r"\bget_ipython\b", line)
            or re.search(r"\bget_iptyhon\b", line)
            or re.search(r"\bmatplotlib\b", line)
            or re.search(r"\bplt\.", line)
        )
        if must_comment and not re.match(r"^\s*#", line):
            out_lines.append("# " + line)
            count += 1
        else:
            out_lines.append(line)
    return "".join(out_lines), count

def remove_palette_colors(text: str) -> tuple[str, int]:
    """
    Remove palette=colors (with optional spaces) and handle commas safely.
    Handles:
      ', palette=colors'  'palette=colors,'  'palette = colors'  ' ,palette = colors ,'
    """
    n_total = 0
    # , palette=colors
    text, n1 = re.subn(r"\s*,\s*palette\s*=\s*colors\b", "", text)
    # palette=colors,
    text, n2 = re.subn(r"\bpalette\s*=\s*colors\s*,\s*", "", text)
    # lone palette=colors (no comma around)
    text, n3 = re.subn(r"\bpalette\s*=\s*colors\b", "", text)
    return text, (n1 + n2 + n3)

def ensure_main_guard(text: str) -> tuple[str, int]:
    if re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]\s*:", text):
        return text, 0
    append_block = "\n\n# Added for safety on macOS multiprocessing\nif __name__ == '__main__':\n    pass\n"
    return text + append_block, 1

def process_file(src: Path, dst: Path, inplace: bool = False):
    text = src.read_text(encoding="utf-8")

    text, n_train  = replace_training_magic(text)
    text, n_loader = replace_make_loader(text)
    text, n_cmt    = comment_selected_lines(text)
    text, n_pal    = remove_palette_colors(text)
    text, n_main   = ensure_main_guard(text)

    if inplace:
        src.write_text(text, encoding="utf-8")
        out_path = src
    else:
        dst.write_text(text, encoding="utf-8")
        out_path = dst

    print(f"[ok] Wrote: {out_path}")
    print(f" - replaced training magic: {n_train}")
    print(f" - replaced make_loader:   {n_loader}")
    print(f" - commented lines:        {n_cmt}")
    print(f" - removed palette=colors: {n_pal}")
    print(f" - added main-guard:       {n_main}")

def main():
    ap = argparse.ArgumentParser(description="Fix converted-notebook .py files.")
    ap.add_argument("input", type=Path, help="Path to input .py file")
    ap.add_argument("-o", "--output", type=Path, help="Path to output .py file (default: <input>.fixed.py)")
    ap.add_argument("--inplace", action="store_true", help="Edit file in place")
    args = ap.parse_args()

    src = args.input
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

    if args.inplace:
        process_file(src, src, inplace=True)
    else:
        out = args.output or src.with_suffix(".fixed.py")
        process_file(src, out, inplace=False)

if __name__ == "__main__":
    main()

