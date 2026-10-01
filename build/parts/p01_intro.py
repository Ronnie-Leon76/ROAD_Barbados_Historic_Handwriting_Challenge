# %% [markdown]
# # R.O.A.D. Barbados Historic Handwriting Challenge — competition pipeline
#
# End-to-end, single-GPU pipeline for transcribing 17th–19th century Barbadian
# deed/will/inventory line crops. Built to be run top-to-bottom on a rented GPU
# (L40S 48 GB / A100 / H100). Every stage is switchable from one `CFG` object.
#
# ## What this pipeline does
#
# 1. **Backbone**: `Qwen3-VL-8B-Instruct` (Apache-2.0) fine-tuned on the challenge's
#    own line images — LoRA by default, or a **full fine-tune** via
#    `CFG.finetune_mode = "full"`, which re-derives the learning rate, optimiser,
#    batch shape and EMA to match. Falls back to `Qwen2.5-VL-7B-Instruct`
#    (Apache-2.0) or `Qwen3-VL-4B-Instruct` on smaller GPUs.
# 2. **Line-aware preprocessing**: ink-cropping, projection-profile deskew, CLAHE,
#    fixed-height aspect-preserving resize aligned to the ViT patch grid so that a
#    12:1 line crop spends its token budget on glyphs instead of white margin.
# 3. **Augmentation**: geometric + photometric + ink morphology, plus a
#    **StackMix-style word-crop recombination** engine that segments words out of the
#    provided training lines and re-renders new labelled lines from them
#    (explicitly permitted by the organisers — see the rules cell below).
# 4. **Training**: bf16 LoRA/QLoRA, FlashAttention-2 when available, gradient
#    checkpointing, **cosine annealing with warmup**, NEFTune, EMA of the adapter
#    weights, and generation-based CER/WER validation driving checkpoint selection.
# 5. **Decoding**: batched beam search producing an N-best list with sequence scores,
#    **multi-scale test-time augmentation**, **ROVER** confusion-network fusion across
#    seeds/scales, and **character n-gram LM rescoring** + lexicon repair, with all
#    fusion hyper-parameters tuned on out-of-fold predictions.
# 6. **GRPO stage** that optimises the competition metric directly as a reward on top
#    of the SFT checkpoint. On by default and self-gating: the polished weights are
#    scored against their parent on held-out data and discarded unless they win.
# 7. **A second, CTC recogniser**: kraken's PP-OCRv6 port (15.9M parameters,
#    Apache-2.0) fine-tuned per fold in its own virtualenv. It cannot loop or
#    hallucinate the way an autoregressive decoder can, so it joins the ROVER vote
#    and anchors the rescoring — and, like GRPO, it is kept only if it wins
#    out-of-fold (`Section 14.1`).
# 8. **Reproducibility and checkpoint custody**: deterministic kernels, per-sample
#    seeding, a run fingerprint over the input files and library versions, and
#    optional mirroring of every checkpoint and artefact to the Hugging Face Hub —
#    so a run can be replayed, or resumed on a fresh pod, to the same score.
# 9. **Unattended operation**: the same cells build `run_pipeline.py`, a headless
#    script that logs to file, streams TensorBoard curves (locally and to the hub),
#    and emails you on every checkpoint upload, GRPO verdict, completion and
#    failure. See `python build/build_script.py` and `./run_background.sh`.
# 10. **Legible progress**: stage banners with elapsed wall clock, a `tqdm` bar on
#    every long loop and a training heartbeat with a projected finish time — throttled
#    automatically when the output is a log file rather than a terminal (`Section 2.6`).
#
# ## Metric
#
# Leaderboard score = `0.5 * (1 - weighted_WER) + 0.5 * (1 - weighted_CER)`, where
# longer references carry more weight. The metric is re-implemented in
# `Section 4` and unit-tested, so local validation numbers are directly comparable
# to the leaderboard.
#
# ## How to actually win (read this before training another 8B fold-0)
#
# The 0.905 public score trained **one fold**. GRPO added 0.0004. First place is
# 0.940. Do not swap to dots.ocr (custom licence) or chase another VLM from the
# BHL printed-book board. Set `CFG.competition_mode = True` (the default): all
# five folds of this Qwen pipeline **and** the kraken CTC branch. Then run
# `ROAD_Barbados_Kraken_PP_OCRv6.ipynb` on a **separate** kernel and submit
# both files plus the ensemble. Email alerts fire in both notebooks from `.env`.

# %% [markdown]
# ## 0. Rules compliance — read before changing model or data settings
#
# These constraints come from the organisers' answers in the challenge discussion
# board and materially restrict what is allowed. The pipeline is built to respect
# them, and `Section 2` programmatically enforces the licence rule.
#
# | Topic | Ruling | How this notebook complies |
# |---|---|---|
# | Pretrained model licences | A model must be openly available **and** its licence must permit commercial use. `stanford-oval/churro-3B` was explicitly **rejected** because of the Qwen Research (non-commercial) licence. | Only Apache-2.0 / MIT backbones are used. A licence guard queries the Hugging Face API at runtime and refuses to load a model whose licence is not on the allow-list. **`Qwen2.5-VL-3B-Instruct` is `qwen-research` and is therefore blocked** — only the 7B (Apache-2.0) member of that family is usable. |
# | StackMix-style augmentation | **Permitted**: segmenting crops from the provided training images and programmatically recombining them into new labelled training examples counts as training-data augmentation, provided the process is fully automated and reproducible. No test images, no external image data, no manual annotation. | `Section 7` implements exactly this: word crops are segmented from `Train.csv` images only, by code, and recombined with labels formed by concatenating the source words. Seeded and reproducible. |
# | External datasets | Not allowed. | No external image corpora. The only text resource used by the LM rescorer is the challenge's own `Train.csv` targets. |
# | Test-time pseudo-labelling / self-training | The organiser thread on this was still open at time of writing. | Implemented in `Section 14` but **disabled by default** (`CFG.use_pseudo_labels = False`). Do not enable it without an explicit organiser ruling. |
# | Reproducibility | Required. | `Section 2.2`: pinned dependency versions, deterministic cuDNN/cuBLAS kernels, every stochastic stage (splits, augmentation, StackMix) seeded per sample rather than per stream, and a run fingerprint that hashes the input CSVs so a replay can be verified rather than assumed. |

# %% [markdown]
# ## 1. Environment
#
# Set `RUN_INSTALL = True` on a fresh pod. The pins below are the combination that
# supports `Qwen3-VL` (needs `transformers >= 4.57`) together with TRL's
# vision-language GRPO trainer.
#
# On RunPod choose a **PyTorch 2.6+ / CUDA 12.4+** template. If the base image ships
# an older torch, the cell prints a warning instead of silently producing a broken
# environment. FlashAttention-2 is installed from a prebuilt wheel with the pip
# cache disabled so a `/tmp` vs `/workspace` split cannot raise EXDEV; if no
# matching wheel exists the model falls back to SDPA and training still runs.
#
# To mirror checkpoints to the Hugging Face Hub, export a **write** token in the
# pod environment (or gitignored `.env`) *before* running `Section 2.3`. Do not
# paste it into a notebook cell — that is how tokens end up in checkpoints and
# git history. Section 2.3 asserts the token is present and sets the repo id.

# %%
import os as _os

# A fresh pod needs the installs. `run_pipeline.py` sets ROAD_HEADLESS=1 because an
# unattended run should not reinstall packages on every launch; pass --install to
# that script on the first run of a new pod.
RUN_INSTALL = _os.environ.get("ROAD_HEADLESS") != "1"
INSTALL_FLASH_ATTENTION = RUN_INSTALL   # falls back to SDPA if the wheel fails

if RUN_INSTALL:
    import subprocess
    import sys
    from pathlib import Path

    def pip(*args: str, extra_env: dict | None = None, cwd: str | None = None) -> int:
        print("pip", *args)
        env = None
        if extra_env:
            env = _os.environ.copy()
            env.update(extra_env)
        return subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *args],
            check=False, env=env, cwd=cwd,
        ).returncode

    pip("--upgrade", "pip")
    pip(
        "transformers==4.57.1",
        "accelerate>=1.10.0",
        "peft>=0.17.0",
        "trl==0.23.0",
        "datasets>=3.0.0",
        "bitsandbytes>=0.47.0",
        "sentencepiece",
        "safetensors",
        "huggingface_hub>=0.34.0",
    )
    pip(
        "opencv-python-headless>=4.9",
        "pillow>=10.2",
        "rapidfuzz>=3.9",
        "jiwer>=3.0",
        "scikit-learn>=1.4",
        "scipy>=1.11",
        "matplotlib>=3.8",
        "pandas>=2.1",
        "tqdm",
        "pyyaml",
        "tensorboard>=2.16",
    )

    def _flash_attn_ok() -> bool:
        try:
            import flash_attn  # noqa: F401
            return True
        except Exception:
            return False

    def _download_flash_attn_wheel(dest_dir: Path) -> Path | None:
        """Fetch the same GitHub wheel flash-attn's setup.py would have guessed."""
        try:
            import torch
            import urllib.request
        except Exception:
            return None
        if sys.platform != "linux":
            return None
        py = f"cp{sys.version_info.major}{sys.version_info.minor}"
        torch_mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
        try:
            abi = "TRUE" if bool(torch._C._GLIBCXX_USE_CXX11_ABI) else "FALSE"
        except Exception:
            abi = "TRUE"
        for ver in ("2.8.3.post1", "2.8.3", "2.8.2", "2.7.4.post1", "2.7.3"):
            name = (f"flash_attn-{ver}+cu12torch{torch_mm}cxx11abi{abi}"
                    f"-{py}-{py}-linux_x86_64.whl")
            url = ("https://github.com/Dao-AILab/flash-attention/releases/"
                   f"download/v{ver}/{name}")
            path = dest_dir / name.replace("+", "-")
            print(f"[env] trying {url}")
            try:
                urllib.request.urlretrieve(url, path)
                if path.stat().st_size > 1_000_000:
                    return path
            except Exception as exc:
                print(f"[env]   {exc}")
            path.unlink(missing_ok=True)
        return None

    def install_flash_attn() -> None:
        """Install a prebuilt wheel without crossing filesystems.

        flash-attn downloads a matching GitHub wheel, then pip `os.rename`s it
        into its cache. On rented pods /tmp and /workspace are often different
        mounts, which raises EXDEV (errno 18) and looks like a failed source
        build even though a good wheel was just fetched. Keep temp files on the
        workspace, skip the pip wheel cache, and never fall through to a CUDA
        compile (~40 minutes on a pod).
        """
        if _flash_attn_ok():
            print("[env] flash-attn already installed")
            return
        root = Path.cwd() / ".cache"
        tmp = root / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        env = {
            "TMPDIR": str(tmp), "TEMP": str(tmp), "TMP": str(tmp),
            "FLASH_ATTENTION_SKIP_CUDA_BUILD": "TRUE",
        }
        rc = pip("flash-attn>=2.7.0", "--no-build-isolation", "--no-cache-dir",
                 extra_env=env, cwd=str(tmp))
        if rc == 0 or _flash_attn_ok():
            print("[env] flash-attn installed")
            return
        wheel = _download_flash_attn_wheel(tmp)
        if wheel is not None and (pip(str(wheel), extra_env=env, cwd=str(tmp)) == 0
                                  or _flash_attn_ok()):
            print(f"[env] flash-attn installed from {wheel.name}")
            return
        print("[env] flash-attn unavailable; attention will use SDPA. "
              "Training still runs, just uses more VRAM per step.")

    if INSTALL_FLASH_ATTENTION:
        install_flash_attn()

# %%
import gc
import json
import math
import os
import random
import re
import shutil
import sys
import time
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
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import torch

print(f"python           {sys.version.split()[0]}")
print(f"torch            {torch.__version__}")
print(f"cuda available   {torch.cuda.is_available()}")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_NAME, GPU_VRAM_GB, N_GPU = "cpu", 0.0, 0

if torch.cuda.is_available():
    N_GPU = torch.cuda.device_count()
    props = torch.cuda.get_device_properties(0)
    GPU_NAME = props.name
    GPU_VRAM_GB = props.total_memory / 1024 ** 3
    print(f"cuda runtime     {torch.version.cuda}")
    print(f"gpu              {N_GPU} x {GPU_NAME} ({GPU_VRAM_GB:.0f} GB, sm_{props.major}{props.minor})")
    print(f"bf16 supported   {torch.cuda.is_bf16_supported()}")
else:
    print("!! No GPU detected. Data/EDA/metric cells run fine; training will not.")

_torch_major_minor = tuple(int(x) for x in torch.__version__.split(".")[:2])
if _torch_major_minor < (2, 6):
    print(
        f"!! torch {torch.__version__} is older than 2.6 - Qwen3-VL kernels and "
        "flash-attn wheels expect >=2.6. Pick a newer pod template if training fails."
    )

# Ampere and newer: allow TF32 for the non-matmul-critical ops and enable the
# fast bf16 reduction paths. Costs nothing in accuracy for this workload.
# `cudnn.benchmark` is set here for the cache-warming pass only; Section 2.2 takes
# it over once CFG.deterministic is known.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def vram_report(tag: str = "") -> None:
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024 ** 3
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    print(f"[vram{(' ' + tag) if tag else ''}] allocated {alloc:.1f} GB | peak {peak:.1f} GB / {GPU_VRAM_GB:.0f} GB")
