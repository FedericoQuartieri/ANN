#!/usr/bin/env python3
"""
Strip lines and blocks from Python files.

Removes ONLY lines containing:
  - get_ipython(
  - google.colab
  - matplotlib
  - plt   (as a whole word, e.g., 'plt.plot', 'as plt')

AND removes entire blocks guarded by:
  - if (not CONVERTIBLE):
  - if not CONVERTIBLE:

Behavior for 'if not CONVERTIBLE' blocks:
  - The whole block is deleted.
  - If the next non-blank line is an 'else:' at the same indentation, the 'else:'
    header is removed and its block is KEPT, dedented to the 'if' level.

Outputs <input_stem>_clean.py by default (no '.py.clean').
"""

import argparse
import re
from pathlib import Path

# -------- line filters --------
PATTERNS = [
    re.compile(r"\bget_ipython\s*\("),  # get_ipython(...)
    re.compile(r"\bgoogle\.colab\b"),   # google.colab
    re.compile(r"\bmatplotlib\b"),      # any matplotlib usage/import
    re.compile(r"\bplt\b"),             # 'plt' as a word (matches 'plt.plot', 'as plt', etc.)
]

# -------- block filters --------
IF_NOT_CONVERTIBLE_RE = re.compile(
    r"^\s*if\s*(\(\s*not\s+CONVERTIBLE\s*\)|\s+not\s+CONVERTIBLE)\s*:\s*$"
)
ELSE_RE = re.compile(r"^\s*else\s*:\s*$")

def count_indent_spaces(line: str) -> int:
    """Count indentation as spaces (tabs expanded to 4 spaces)."""
    expanded = line.expandtabs(4)
    return len(expanded) - len(expanded.lstrip(' '))

def dedent_by(line: str, spaces: int) -> str:
    """Dedent a line by 'spaces' (spaces only, tabs are expanded first)."""
    expanded = line.expandtabs(4)
    # remove up to 'spaces' leading spaces (without going below zero)
    to_strip = min(spaces, count_indent_spaces(expanded))
    return expanded[to_strip:]

def strip_if_not_convertible_blocks(text: str) -> tuple[str, int, int]:
    """
    Remove blocks under `if (not CONVERTIBLE):` or `if not CONVERTIBLE:`.
    If an aligned `else:` follows, remove the 'else:' line and keep its block,
    dedented to the 'if' baseline.
    Returns: (new_text, blocks_removed, lines_removed)
    """
    lines = text.splitlines(keepends=True)
    out = []
    i = 0
    blocks_removed = 0
    lines_removed = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        if IF_NOT_CONVERTIBLE_RE.match(line):
            base_indent = count_indent_spaces(line)
            i += 1
            blocks_removed += 1
            lines_removed += 1  # removed the 'if' line

            # remove the whole IF block (all lines with indent > base, plus blank lines)
            while i < n:
                nxt = lines[i]
                if nxt.strip() == "":
                    lines_removed += 1
                    i += 1
                    continue
                if count_indent_spaces(nxt) > base_indent:
                    lines_removed += 1
                    i += 1
                    continue
                break  # block ended (dedent)

            # Check for an 'else:' at same indent; if found, remove header and keep its body dedented
            # skip purely blank separators before else (rare/invalid in Python, but be forgiving)
            j = i
            while j < n and lines[j].strip() == "":
                # We'll drop blank lines between 'if' and 'else' header
                lines_removed += 1
                j += 1

            if j < n and ELSE_RE.match(lines[j]) and count_indent_spaces(lines[j]) == base_indent:
                # remove 'else:' header
                lines_removed += 1
                j += 1

                # skip leading blank lines inside else body (drop them)
                while j < n and lines[j].strip() == "":
                    lines_removed += 1
                    j += 1

                # dedent else-body to base indent and KEEP it
                if j < n:
                    # compute the first meaningful indent in else-body
                    first_indent = count_indent_spaces(lines[j])
                else:
                    first_indent = base_indent + 4  # arbitrary

                delta = max(0, first_indent - base_indent)

                i = j
                while i < n:
                    body_line = lines[i]
                    if body_line.strip() == "":
                        # blank line: keep as-is
                        out.append(body_line)
                        i += 1
                        continue
                    cur_indent = count_indent_spaces(body_line)
                    if cur_indent > base_indent:
                        out.append(dedent_by(body_line, delta))
                        i += 1
                    else:
                        break
                # continue main while
                continue

            # no 'else:' → proceed; i is at first line after the removed block
            continue

        # Normal line: emit for now (line-level filters will run later)
        out.append(line)
        i += 1

    return "".join(out), blocks_removed, lines_removed

def should_drop_line(line: str) -> bool:
    return any(p.search(line) for p in PATTERNS)

def strip_lines(text: str) -> tuple[str, int]:
    removed = 0
    out_lines = []
    for line in text.splitlines(keepends=True):
        if should_drop_line(line):
            removed += 1
            continue
        out_lines.append(line)
    return "".join(out_lines), removed

def process_file(src: Path, dst: Path) -> tuple[int, int, int]:
    """
    Returns (blocks_removed, lines_removed_in_blocks, lines_removed_by_patterns)
    """
    text = src.read_text(encoding="utf-8", errors="ignore")

    # 1) Remove 'if not CONVERTIBLE' blocks (and optional else handling)
    text, blocks_removed, block_lines_removed = strip_if_not_convertible_blocks(text)

    # 2) Remove single lines by patterns
    text, pattern_lines_removed = strip_lines(text)

    dst.write_text(text, encoding="utf-8")
    return blocks_removed, block_lines_removed, pattern_lines_removed

def default_dst(src: Path) -> Path:
    # Always write a Python file named <stem>_clean.py in the same folder
    return src.with_name(f"{src.stem}_clean.py")

def main():
    ap = argparse.ArgumentParser(
        description="Strip plt/matplotlib/get_ipython/google.colab lines and remove `if not CONVERTIBLE:` blocks."
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

        blocks, block_lines, pattern_lines = process_file(src, dst)
        print(f"[ok] {src} -> {dst} | blocks removed: {blocks} | "
              f"lines in blocks: {block_lines} | single lines: {pattern_lines}")

if __name__ == "__main__":
    main()
