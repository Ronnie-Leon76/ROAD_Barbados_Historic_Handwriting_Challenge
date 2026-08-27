"""Assemble the competition notebook from percent-format source parts.

The source of truth lives in ``build/parts/*.py`` (jupytext "percent" format).
Running this script concatenates the parts in filename order and emits a single
executable ``.ipynb``.  Keeping the source as plain Python means the whole
pipeline can be linted, diffed and syntax-checked without Jupyter.

Usage:
    python build/build_notebook.py [--out ROAD_Barbados_HTR_SOTA.ipynb] [--check]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

CELL_MARK = "# %%"


def parse_percent_file(text: str):
    """Split a percent-format file into (cell_type, source) tuples."""
    cells = []
    current_type = None
    current: list[str] = []

    def flush():
        if current_type is None:
            return
        src = "\n".join(current).strip("\n")
        if not src.strip():
            return
        if current_type == "markdown":
            lines = []
            for line in src.split("\n"):
                stripped = line.lstrip()
                if stripped.startswith("# "):
                    lines.append(stripped[2:])
                elif stripped == "#":
                    lines.append("")
                else:
                    lines.append(line)
            src = "\n".join(lines).strip("\n")
        cells.append((current_type, src))

    for line in text.split("\n"):
        if line.startswith(CELL_MARK):
            flush()
            current = []
            header = line[len(CELL_MARK):].strip()
            current_type = "markdown" if header.startswith("[markdown]") else "code"
        else:
            current.append(line)
    flush()
    return cells


def build(parts_dir: pathlib.Path, out_path: pathlib.Path) -> dict:
    part_files = sorted(parts_dir.glob("p*.py"))
    if not part_files:
        raise SystemExit(f"No parts found in {parts_dir}")

    nb_cells = []
    for path in part_files:
        text = path.read_text(encoding="utf-8")
        try:
            compile(text, str(path), "exec")
        except SyntaxError as exc:  # surface the offending part immediately
            raise SystemExit(f"SyntaxError in {path.name}:{exc.lineno}: {exc.msg}") from exc
        for cell_type, src in parse_percent_file(text):
            if cell_type == "markdown":
                nb_cells.append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": src.split("\n"),
                })
            else:
                nb_cells.append({
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": src.split("\n"),
                })

    # nbformat wants each source line to keep its trailing newline except the last
    for cell in nb_cells:
        lines = cell["source"]
        cell["source"] = [ln + "\n" for ln in lines[:-1]] + [lines[-1]]

    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return nb


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=str(root / "parts"))
    ap.add_argument("--out", default=str(root.parent / "ROAD_Barbados_HTR_SOTA.ipynb"))
    args = ap.parse_args()

    nb = build(pathlib.Path(args.parts), pathlib.Path(args.out))
    n_code = sum(c["cell_type"] == "code" for c in nb["cells"])
    n_md = sum(c["cell_type"] == "markdown" for c in nb["cells"])
    n_lines = sum(len(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    print(f"wrote {args.out}: {n_code} code cells, {n_md} markdown cells, {n_lines} code lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
