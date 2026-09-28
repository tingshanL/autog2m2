#!/usr/bin/env python3
"""Generate all figures from the result tables."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PLOTTERS = (
    "plot_figure1.py",
    "plot_figure2.py",
    "plot_figure2_auto_k.py",
    "plot_figure3.py",
    "plot_figure4.py",
    "plot_figure5.py",
    "plot_figure6.py",
    "plot_figure7_tcga.py",
    "plot_figure7_auto_k.py",
    "plot_figure8.py",
    "plot_figure9.py",
)


def main() -> int:
    code_dir = Path(__file__).resolve().parent
    for filename in PLOTTERS:
        print(f"Running {filename}")
        subprocess.run([sys.executable, str(code_dir / filename)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
