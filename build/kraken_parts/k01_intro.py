# %% [markdown]
# # R.O.A.D. Barbados — Kraken PP-OCRv6 pipeline
#
# A **Kraken-first** competition notebook. The Qwen3-VL-8B LoRA run
# (`ROAD_Barbados_HTR_SOTA_Runpod.ipynb`) landed at **0.90491 OOF / 0.90485 public**
# after one fold + GRPO. First place sits at **0.9399**. GRPO added `+0.00041`.
# Post-processing added `+0.00258`. The remaining gap is not a ranking trick —
# the recogniser is still reading ~15 % of words wrong (OOF WER 0.150, CER 0.045).
#
# This notebook switches architecture.
#
# ## Why PP-OCRv6 instead of another VLM
#
# On the BHL historical OCR board (2 165 pages of 18th–19th-century books,
# 2026-09-03) **kraken PP-OCRv6 medium (15.9 M params, Apache-2.0)** reads at
# CER 0.034 vs 0.051 for Qwen3.5-9B, with **0 % looped pages**. It is a CTC line
# model: one label per image column, so it cannot invent a second copy of a
# legal formula the way an autoregressive decoder can. The challenge labels are
# diplomatic transcriptions of secretary hand, which is exactly the setting
# where "literal CTC" beats "fluent VLM".
#
# The starter-kit Kraken script is **not** this. It page-segments already-cropped
# lines, binarises them, and continues a 2024 CATMuS model for one epoch. This
# notebook:
#
# 1. Treats every crop as **one line** (no page segmentation).
# 2. Fine-tunes `small-models-for-glam/kraken-ppocrv6-medium` with `--resize new`
#    so the 44-language softmax collapses onto the challenge charset.
# 3. Trains **all five group-aware folds** (16 M params, minutes per fold).
# 4. Trains the **small** sibling as a second CTC family (cheap diversity).
# 5. Decodes with photometric TTA, then ROVER + char-LM + lexicon, all tuned OOF.
# 6. Optionally **ensembles the Qwen submission** when it is on disk — the two
#    families fail differently, which is what a confusion-network vote converts
#    into points.
# 7. Emails you on start, each fold, and finish (and the traceback on a crash),
#    using the same `.env` SMTP settings as the Qwen notebook.
#
# ## BHL board vs this challenge — do not pick one row
#
# The BHL table is **zero-shot printed books**. This challenge is **fine-tuned
# secretary-hand line crops**. Rankings do not transfer:
#
# | Model | Why it is (or is not) the move here |
# |---|---|
# | **Qwen3-VL-8B, fine-tuned** | Already 0.905 after **one** fold. The unused lever is the other four folds, not a different VLM. |
# | **kraken PP-OCRv6** | Best *second* member: CTC cannot loop; Apache-2.0; 16 M params so 5 folds are cheap. |
# | dots.ocr / dots.mocr | Top of BHL, but the weights sit under a **custom** licence agreement, not a clean Apache/MIT. The organisers DQ'd churro-3B for exactly that class of problem. Do not submit it. |
# | PaddleOCR-VL-1.6 | Apache-2.0, but a page/document VLM. This dataset is already line-cropped. |
# | OvisOCR2 | Strong specialist; licence is not on the commercial allow-list we can defend. |
#
# The submission that can actually reach ~0.94 is **5-fold Qwen + 5-fold Kraken,
# ROVER-fused**. Run this notebook and the Qwen notebook with `competition_mode`
# (all folds). Submit both singles and the ensemble; keep the public-board winner.
#
# **Do not run this inside the Qwen kernel.** kraken 7.1 pins `torch>=2.9` and
# will fight that environment. A fresh venv / pod template is the whole setup.
#
# ## Metric
#
# `0.5 * (1 - weighted_WER) + 0.5 * (1 - weighted_CER)`, length-power `0.5`,
# copied from the VLM notebook and unit-tested. Empty predictions score 0.
#
# ## Rules
#
# | Topic | Ruling | This notebook |
# |---|---|---|
# | Model licence | Must permit commercial use | `kraken-ppocrv6-{medium,small}` are Apache-2.0; a licence guard refuses anything else |
# | StackMix | Permitted on Train.csv images only | Optional photometric extra views; no test images |
# | External data | Not allowed | Fine-tune uses Train.csv only |
# | Self-training on test | Unresolved | Disabled |

# %%
import os as _os
import subprocess
import sys

RUN_INSTALL = _os.environ.get("ROAD_HEADLESS") != "1"

if RUN_INSTALL:
    def pip(*args: str) -> int:
        print("pip", *args)
        return subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args],
                              check=False).returncode

    pip("--upgrade", "pip")
    pip("huggingface_hub>=0.34.0", "opencv-python-headless>=4.9", "pillow>=10.2",
        "rapidfuzz>=3.9", "scikit-learn>=1.4", "pandas>=2.1", "tqdm", "pyyaml",
        "tensorboard>=2.16")
    # kraken 7.1.1 is the first release that trains the PP-OCRv6 architecture.
    rc = pip("kraken==7.1.1")
    if rc != 0:
        print("[env] kraken==7.1.1 failed; trying kraken>=7.1.0")
        pip("kraken>=7.1.0")

# %%
import gc
import json
import math
import os
import random
import re
import shutil
import time
import unicodedata
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

print(f"python           {sys.version.split()[0]}")
print(f"torch            {torch.__version__}")
print(f"cuda available   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    GPU_NAME = torch.cuda.get_device_name(0)
    GPU_VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"gpu              {GPU_NAME} ({GPU_VRAM_GB:.1f} GB)")
else:
    GPU_NAME, GPU_VRAM_GB = "cpu", 0.0
    print("gpu              none - ketos will train on CPU (slow, but 16M params fit)")

try:
    import kraken
    print(f"kraken           {getattr(kraken, '__version__', 'unknown')}")
except Exception as exc:
    print(f"!! kraken import failed ({exc}). Re-run the install cell in a fresh kernel.")
