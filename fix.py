#!/usr/bin/env python3
"""
Remove ONLY lines that contain any of:
  - get_ipython(
  - google.colab
  - matplotlib
  - plt   (as a whole word, e.g., 'plt.plot', 'as plt')

Outputs <input_stem>_clean.py by default (no more '.py.clean').
"""

import argparse
import re
from pathlib import Path

PATTERNS = [
    re.compile(r"\bget_ipython\s*\("),  # get_ipython(...)
    re.compile(r"\bgoogle\.colab\b"),   # google.colab
    re.compile(r"\bmatplotlib\b"),      # any matplotlib usage/import
    re.compile(r"\bplt\b"),             # 'plt' as a word (matches 'plt.plot', 'as plt', etc.)
]

def should_drop(line: str) -> bool:
    return any(p.search(line) for p in PATTERNS)

def process_file(src: Path, dst: Path) -> tuple[int, int]:
    text = src.read_text(encoding="utf-8", errors="ignore")
    out_lines = []
    removed = 0
    for line in text.splitlines(keepends=True):
        if should_drop(line):
            removed += 1
            continue
        out_lines.append(line)
    dst.write_text("".join(out_lines), encoding="utf-8")
    kept = len(text.splitlines()) - removed
    return removed, kept

def default_dst(src: Path) -> Path:
    # Always write a Python file named <stem>_clean.py in the same folder
    return src.with_name(f"{src.stem}_clean.py")

def main():
    ap = argparse.ArgumentParser(
        description="Strip lines containing plt/matplotlib/get_ipython/google.colab"
    )
    ap.add_argument("inputs", type=Path, nargs="+", help="Input .py files")
    ap.add_argument("-o", "--output", type=Path, help="Output file (only with one input)")
    ap.add_argument("--inplace", action="store_true", help="Overwrite input file(s)")
    args = ap.parse_args()

    if args.output and len(args.inputs) != 1:
        ap.error("-o/--output can be used only with a single input file")

    for src in args.inputs:
        if not src.exists():
            print(f"[skip] Not found: {src}")
            continue

        if args.inplace:
            dst = src
        elif args.output:
            dst = args.output
        else:
            dst = default_dst(src)

        removed, kept = process_file(src, dst)
        print(f"[ok] {src} -> {dst} | removed: {removed} line(s), kept: {kept}")

if __name__ == "__main__":
    main()
