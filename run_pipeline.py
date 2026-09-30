#!/usr/bin/env python
"""R.O.A.D. Barbados Historic Handwriting Challenge - unattended training run.

GENERATED FILE - do not edit. Regenerate with:

    python build/build_script.py

Source of truth: build/parts/p*.py (the same cells as ROAD_Barbados_HTR_SOTA.ipynb).

Examples:
    python run_pipeline.py                                  # all defaults
    python run_pipeline.py --folds 0 1 2 3 4 --epochs 5
    python run_pipeline.py --mode full --model Qwen/Qwen3-VL-4B-Instruct
    python run_pipeline.py --hf-repo me/road-htr --no-grpo
    python run_pipeline.py --no-ctc                         # VLM only
    python run_pipeline.py --set lora_r=64 --set line_height=192

Background:
    nohup python -u run_pipeline.py --folds 0 1 2 3 4 > logs/run.out 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # headless: figures are saved, never shown


def _parse_value(raw: str):
    """Coerce a --set value into bool/int/float/tuple, falling back to str."""
    low = raw.strip().lower()
    if low in {"true", "false"}:
        return low == "true"
    if low in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            pass
    if "," in raw:
        return tuple(_parse_value(p) for p in raw.split(","))
    return raw


def _args():
    ap = argparse.ArgumentParser(
        description="Train and submit, unattended.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--folds", type=int, nargs="+", metavar="N",
                    help="folds to train, e.g. --folds 0 1 2 3 4")
    ap.add_argument("--mode", choices=("lora", "full"), help="fine-tuning mode")
    ap.add_argument("--model", metavar="ID", help="backbone model id")
    ap.add_argument("--epochs", type=float)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--hf-repo", metavar="ID", help="hub repo for checkpoints")
    ap.add_argument("--email", metavar="ADDR", help="override NOTIFY_EMAIL_TO")
    ap.add_argument("--grpo", dest="grpo", action="store_true", default=None)
    ap.add_argument("--no-grpo", dest="grpo", action="store_false")
    ap.add_argument("--no-stackmix", dest="stackmix", action="store_false", default=None)
    ap.add_argument("--no-ctc", dest="ctc", action="store_false", default=None,
                    help="skip the kraken PP-OCRv6 CTC member (Section 14.1)")
    ap.add_argument("--no-tensorboard", dest="tensorboard", action="store_false",
                    default=None)
    ap.add_argument("--no-email", dest="notify", action="store_false", default=None)
    ap.add_argument("--resume-from-hub", action="store_true",
                    help="download checkpoints instead of training")
    ap.add_argument("--fast", action="store_true",
                    help="1 fold, 1 epoch, greedy decoding - a plumbing check")
    ap.add_argument("--install", action="store_true",
                    help="pip install the pinned dependencies first (new pod)")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--progress-interval", type=float, default=30.0, metavar="SEC",
                    help="seconds between progress-bar refreshes in the log")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="set any CFG field, repeatable")
    return ap.parse_args()


class _Tee:
    """Mirror stdout/stderr to a log file so a detached run stays inspectable.

    Progress bars redraw with a carriage return, which a terminal renders in place
    but a file would accumulate into one enormous line. Treating \r as a line break
    turns every bar refresh into its own timestamped log entry instead, so the log
    reads as a timeline. Section 2.6 throttles those refreshes to one per 30s when
    the output is not a terminal, which is what keeps the volume sane.
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.file = open(path, "a", buffering=1, encoding="utf-8", errors="replace")
        self._fresh = True

    def write(self, data):
        self.stream.write(data)
        for i, chunk in enumerate(data.replace("\r\n", "\n").replace("\r", "\n").split("\n")):
            if i:
                self.file.write("\n")
                self._fresh = True
            if chunk:
                if self._fresh:
                    self.file.write(time.strftime("%H:%M:%S "))
                    self._fresh = False
                self.file.write(chunk)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def isatty(self):
        return False


def _bootstrap_crash_handler(enabled: bool = True):
    """Email uncaught exceptions from before the pipeline's own notifier exists.

    Startup failures - a missing CSV, a bad hub token, an import error from a
    half-installed environment - happen before Section 2.4 runs. Without this they
    would only ever appear in the log file of a machine nobody is watching. Section
    2.4 replaces this hook once the real Notifier is constructed.
    """
    import smtplib
    import traceback
    from email.message import EmailMessage

    if not enabled:
        return

    def load_env():
        path = Path(".env")
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))

    load_env()

    def hook(exc_type, exc, tb):
        detail = "".join(traceback.format_exception(exc_type, exc, tb))
        to = os.environ.get("NOTIFY_EMAIL_TO", "")
        user = os.environ.get("SMTP_USER", "")
        password = os.environ.get("SMTP_PASSWORD", "")
        host = os.environ.get("SMTP_HOST", "")
        if to and user and password and host:
            try:
                msg = EmailMessage()
                msg["Subject"] = "[ROAD-HTR] FAILURE during start-up"
                msg["From"], msg["To"] = user, to
                msg.set_content(
                    f"The run failed before training began.\n\n{detail}"[:8000])
                with smtplib.SMTP(host, int(os.environ.get("SMTP_PORT", 587)),
                                  timeout=30) as server:
                    server.starttls()
                    server.login(user, password)
                    server.send_message(msg)
                print(f"[notify] start-up failure emailed to {to}")
            except Exception as mail_exc:
                print(f"[notify] could not report the failure: {mail_exc}")
        sys.__excepthook__(exc_type, exc, tb)

    sys.excepthook = hook


ARGS = _args()
_bootstrap_crash_handler(enabled=ARGS.notify is not False)

# Tells the environment cell not to reinstall packages on every launch.
os.environ["ROAD_HEADLESS"] = "0" if ARGS.install else "1"
os.environ.setdefault("PROGRESS_INTERVAL", str(ARGS.progress_interval))

# ------------------------------------------------------------------ logging --
LOG_DIR = Path(ARGS.log_dir)
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.log"
sys.stdout = _Tee(sys.stdout, LOG_PATH)
sys.stderr = _Tee(sys.stderr, LOG_PATH)
print(f"[script] logging to {LOG_PATH}")
print(f"[script] pid {os.getpid()}")

# ------------------------------------------------- config overrides from argv --
CLI_OVERRIDES: dict = {}
if ARGS.folds is not None:
    CLI_OVERRIDES["train_folds"] = tuple(ARGS.folds)
for flag, field in (("mode", "finetune_mode"), ("model", "model_id"),
                    ("epochs", "epochs"), ("lr", "lr"), ("seed", "seed"),
                    ("hf_repo", "hf_repo_id"), ("grpo", "run_grpo"),
                    ("stackmix", "stackmix"), ("ctc", "ctc_branch"),
                    ("tensorboard", "tensorboard"),
                    ("notify", "notify_email")):
    value = getattr(ARGS, flag, None)
    if value is not None:
        CLI_OVERRIDES[field] = value
if ARGS.resume_from_hub:
    CLI_OVERRIDES["hf_resume"] = True
if ARGS.fast:
    CLI_OVERRIDES.update({"train_folds": (0,), "epochs": 1.0, "num_beams": 1,
                          "n_best": 1, "tta_scales": (1.0,), "tta_photometric": False,
                          "run_grpo": False, "eval_max_samples": 64,
                          "ctc_folds": (0,), "ctc_epochs": 2,
                          "ctc_tta_modes": ("identity",)})
for item in ARGS.set:
    if "=" not in item:
        raise SystemExit(f"--set expects FIELD=VALUE, got {item!r}")
    key, _, raw = item.partition("=")
    CLI_OVERRIDES[key.strip()] = _parse_value(raw)
if ARGS.email:
    os.environ["NOTIFY_EMAIL_TO"] = ARGS.email

# ==========================================================================
# Below this line: the notebook cells, in order, generated from build/parts.
# ==========================================================================

# ==========================================================================
# p01_intro.py
# ==========================================================================

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
# To mirror checkpoints to the Hugging Face Hub, set a **write** token before running
# `Section 2.3` — on RunPod the cleanest way is a pod secret, so the token never
# reaches the notebook file:
#
# ```python
# import os; os.environ["HF_TOKEN"] = "hf_..."   # or set it in the pod environment
# CFG.hf_repo_id = "your-name/road-barbados-htr"
# ```

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

# ==========================================================================
# p02_config.py
# ==========================================================================

# ## 2. Configuration
#
# One object controls the whole run. Defaults are auto-scaled to the detected GPU:
# an 8B backbone in bf16 on >= 40 GB cards, a 4B backbone or 4-bit QLoRA on smaller
# ones. Every downstream cell reads from `CFG`, so a single edit here is enough to
# change the experiment.
#
# Three switches change the shape of the run rather than a hyperparameter:
#
# | Switch | Effect |
# |---|---|
# | `finetune_mode` | `"lora"` adapters (default) or `"full"` — every weight trained, with the learning rate, optimiser, batch shape and EMA re-derived to match |
# | `run_grpo` | RL polish that optimises the leaderboard metric directly, kept only if it beats the SFT adapter out-of-fold |
# | `hf_repo_id` | mirrors every checkpoint and artefact to the Hugging Face Hub so the run can be replayed exactly |

def load_dotenv(path: str | Path = ".env") -> list:
    """Minimal .env loader. Existing environment variables always win.

    Secrets (hub token, SMTP password) are read from the environment so that the
    notebook itself stays shareable. `.env` is gitignored.
    """
    path = Path(path)
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and val and not os.environ.get(key):
            os.environ[key] = val
            loaded.append(key)
    return loaded


_ENV_KEYS = load_dotenv()
print(f"[env] loaded {len(_ENV_KEYS)} settings from .env: {', '.join(_ENV_KEYS) or '(none)'}")


@dataclass
class Config:
    # ---------------------------------------------------------------- paths --
    work_dir: str = "."
    data_dir: str = "data"
    image_subdir: str = "images"          # resolved after unzip, may be nested
    out_dir: str = "artifacts"
    train_csv: str = "Train.csv"
    test_csv: str = "Test.csv"
    sample_sub_csv: str = "SampleSubmission.csv"
    images_url: str = "https://storage.googleapis.com/road-handwriting/images.zip"

    # ---------------------------------------------------------------- model --
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    load_in_4bit: bool = False
    attn_implementation: str = "auto"      # auto -> flash_attention_2 if importable
    # Longest training label is 120 characters, i.e. ~35 Qwen tokens. Keeping this
    # tight matters: beam search with output_scores keeps one logit tensor per step,
    # so decode-time memory is linear in max_new_tokens x beams x batch.
    max_new_tokens: int = 64

    # ------------------------------------------------------------ image i/o --
    patch_grid: int = 32                   # ViT patch * spatial merge, see Section 6
    line_height: int = 160                 # target line height after normalisation
    max_line_width: int = 3200             # p90 aspect is 23.7 -> 3200 covers most
    min_line_width: int = 160
    min_line_height: int = 64
    max_upscale: float = 2.5               # never invent detail beyond this factor
    ink_crop_horizontal: bool = True        # trim blank left/right margins only
    ink_crop_vertical: bool = False         # DANGEROUS: crops carry neighbour lines
    deskew: bool = True
    deskew_max_deg: float = 3.0
    clahe: bool = True
    background_flatten: bool = True         # divide out the parchment illumination

    # --------------------------------------------------------------- splits --
    n_folds: int = 5
    train_folds: tuple = (0,)              # which folds to actually train
    fold_seed: int = 20260703

    # ------------------------------------------------------------- training --
    # "lora"  - rank-decomposition adapters, ~0.5% of parameters trainable.
    # "full"  - every weight updated. Needs roughly 16x the VRAM of LoRA for the
    #           same backbone, so it is only reachable on the 4B model (or on 80 GB
    #           cards) and it overfits 3 300 lines quickly. See Section 11.1.
    finetune_mode: str = "lora"
    seed: int = 42
    deterministic: bool = True             # cuDNN/cuBLAS determinism, see Section 2.2
    epochs: float = 5.0
    lr: float = 1.2e-4
    min_lr_ratio: float = 0.05             # cosine floor, see scheduler cell
    warmup_ratio: float = 0.04
    weight_decay: float = 0.01
    max_grad_norm: float = 0.5
    lr_scheduler_type: str = "cosine"      # cosine | cosine_with_restarts | linear
    per_device_batch_size: int = 8
    grad_accum: int = 4                    # effective batch = bs * accum
    optim: str = "adamw_torch_fused"
    neftune_alpha: float = 5.0             # 0 disables
    label_smoothing: float = 0.0
    ema_decay: float = 0.999               # 0 disables adapter EMA
    evals_per_epoch: float = 1.0
    eval_max_samples: int = 320            # generation-based validation subset
    group_by_length: bool = True
    dataloader_workers: int = 8

    # ----------------------------------------------------------------- lora --
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    use_rslora: bool = True
    use_dora: bool = False                 # ~25% slower, occasionally worth it
    train_vision_tower: bool = True        # LoRA on the ViT blocks too
    train_merger: bool = True              # LoRA on the vision->LLM projector

    # ------------------------------------------------------- full fine-tune --
    # Only read when finetune_mode == "full". The learning rate is two orders of
    # magnitude below the LoRA one: full updates move every weight, and the peak
    # LoRA rate would erase the pretrained language prior in a few hundred steps.
    full_lr: float = 1e-5
    full_epochs: float = 3.0
    full_freeze_vision: bool = False       # freeze the ViT, train the LLM + merger
    full_freeze_llm: bool = False          # the mirror image; useful on small cards

    # ----------------------------------------------------------- augmentation --
    aug_prob: float = 0.9
    aug_rotate_deg: float = 1.5
    aug_shear: float = 0.06
    aug_scale_jitter: float = 0.08
    aug_elastic_prob: float = 0.35
    aug_photometric_prob: float = 0.7
    aug_morph_prob: float = 0.3
    aug_noise_prob: float = 0.3
    aug_cutout_prob: float = 0.2

    # -------------------------------------------------------------- stackmix --
    # Word-crop recombination. Only ~7% of lines segment cleanly enough to trust
    # (measured in Section 7), so the bank is small and the ratio is conservative.
    # A/B it by training fold 0 twice, once with this False, before a long run.
    stackmix: bool = True
    stackmix_ratio: float = 0.30           # synthetic lines per real line, per epoch
    stackmix_min_words: int = 8
    stackmix_max_words: int = 18
    stackmix_ngram_prob: float = 0.7       # real corpus n-grams vs. Markov samples
    stackmix_margin: float = 1.5           # gap-separation quality gate
    stackmix_corr: float = 0.75            # width/word-length correlation gate
    stackmix_seed: int = 1234

    # ------------------------------------------------------------- inference --
    infer_batch_size: int = 16
    num_beams: int = 5
    n_best: int = 5
    length_penalty: float = 1.0
    tta_scales: tuple = (1.0, 1.22)
    tta_photometric: bool = True

    # ---------------------------------------------------------------- fusion --
    # ROVER / rescoring / repair switches are chosen by the search in Section 15
    # and stored in `POST`; only the search budget lives here.
    lm_order: int = 7
    lm_tune_iters: int = 400

    # ------------------------------------------------------- optional stages --
    # GRPO optimises the leaderboard metric directly. It is on, but gated: the
    # polished adapter is scored against the SFT one on held-out data and is only
    # kept if it wins, so enabling it cannot make the submission worse.
    run_grpo: bool = True
    grpo_steps: int = 300
    grpo_num_generations: int = 6
    grpo_lr: float = 2e-6
    grpo_beta: float = 0.02                # KL leash back to the SFT policy
    grpo_temperature: float = 0.9
    grpo_keep_if_better: bool = True       # revert to the SFT adapter when it loses
    grpo_eval_samples: int = 320
    use_pseudo_labels: bool = False        # OFF pending an organiser ruling

    # ------------------------------------------------- kraken CTC branch --
    # A second, architecturally different recogniser: kraken's PP-OCRv6 port, a
    # 15.9M-parameter CTC line model (Apache-2.0), fine-tuned per fold on the same
    # lines. CTC cannot loop or invent text the way an autoregressive decoder can,
    # so it is used as an independent ROVER member and as an anchor that
    # penalises VLM hypotheses which drift away from what is on the page. Section
    # 15 tunes the fusion with and without it and keeps it only if it wins
    # out-of-fold - enabling it cannot make the submission worse. See Section 14.1.
    ctc_branch: bool = True
    ctc_model_id: str = "small-models-for-glam/kraken-ppocrv6-medium"
    ctc_model_file: str = "medium.safetensors"
    ctc_kraken_version: str = "7.1.1"      # first release that trains ppocrv6
    ctc_venv: str = ".venv_kraken"         # isolated: kraken pins torch/scipy/sklearn
    ctc_torch_index: str = "https://download.pytorch.org/whl/cu126"
    ctc_finetune: bool = True              # False = zero-shot base model only
    ctc_folds: tuple = ()                  # () -> every fold; ~minutes each
    ctc_epochs: int = 40
    ctc_lr: float = 3e-4                   # base recipe used 1.05e-3 from scratch
    ctc_warmup: int = 200                  # steps
    ctc_batch_size: int = 32
    ctc_line_height: int = 128             # the medium model's input height
    # ketos height-normalises every line to ctc_line_height and silently DROPS
    # training lines wider than --max-width afterwards; batches are padded to it.
    # 0 = derive it from the data so no line is lost (Section 14.1 prints it).
    ctc_max_width: int = 0
    ctc_augment: bool = True
    ctc_tta_modes: tuple = ("identity", "dark", "light")
    ctc_infer_batch: int = 64

    # ------------------------------------------------------------ hugging face --
    # Set hf_repo_id to something like "your-name/road-barbados-htr" and export a
    # write token as $HF_TOKEN. Everything needed to reproduce a score - adapters,
    # tuned post-processing, manifest, submission - is then mirrored to the hub.
    hf_repo_id: str = ""
    hf_private: bool = True
    hf_push_checkpoints: bool = True
    hf_push_submission: bool = True
    hf_resume: bool = False                # pull checkpoints instead of training
    hf_token_env: str = "HF_TOKEN"

    # -------------------------------------------------------------- monitoring --
    tensorboard: bool = True               # scalar curves, mirrored to the hub
    notify_email: bool = True              # needs SMTP_* + NOTIFY_EMAIL_TO in .env

    def path(self, *parts: str) -> Path:
        return Path(self.work_dir).joinpath(*parts)


CFG = Config()

# --------------------------------------------------------------------------
# Auto-scale the heavy knobs to the GPU we actually got.
# --------------------------------------------------------------------------
def autoscale(cfg: Config, vram_gb: float) -> Config:
    if vram_gb >= 75:            # H100 80GB / H200
        cfg.model_id = "Qwen/Qwen3-VL-8B-Instruct"
        cfg.load_in_4bit = False
        cfg.per_device_batch_size, cfg.grad_accum = 16, 2
        cfg.infer_batch_size = 32
    elif vram_gb >= 40:          # L40S 48GB / A100 40-80GB / RTX 6000
        cfg.model_id = "Qwen/Qwen3-VL-8B-Instruct"
        cfg.load_in_4bit = False
        cfg.per_device_batch_size, cfg.grad_accum = 8, 4
        cfg.infer_batch_size = 16
    elif vram_gb >= 22:          # 4090 / A10 / L4
        cfg.model_id = "Qwen/Qwen3-VL-4B-Instruct"
        cfg.load_in_4bit = True
        cfg.per_device_batch_size, cfg.grad_accum = 4, 8
        cfg.infer_batch_size = 8
        cfg.line_height, cfg.max_line_width = 128, 2560
    elif vram_gb > 0:
        cfg.model_id = "Qwen/Qwen3-VL-4B-Instruct"
        cfg.load_in_4bit = True
        cfg.per_device_batch_size, cfg.grad_accum = 2, 16
        cfg.infer_batch_size = 4
        cfg.line_height, cfg.max_line_width = 96, 2048
        cfg.lora_r, cfg.lora_alpha = 32, 64
    return cfg


def infer_patch_grid(model_id: str) -> int:
    """Pixels covered by one visual token: ViT patch size x spatial merge factor."""
    if "Qwen3-VL" in model_id:
        return 32          # patch 16 x merge 2
    if "Qwen2.5-VL" in model_id or "Qwen2-VL" in model_id:
        return 28          # patch 14 x merge 2
    return 32


def params_billions(model_id: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)B", model_id)
    return float(m.group(1)) if m else 8.0


def apply_finetune_mode(cfg: Config, vram_gb: float) -> Config:
    """Re-derive the knobs that a full fine-tune cannot share with LoRA.

    The three that matter are the learning rate (100x lower), the optimiser state
    (8-bit, because fp32 Adam moments for an 8B model are 64 GB on their own) and
    the EMA, whose fp32 shadow copy is the size of the trainable set - fine for
    LoRA's ~100M parameters, impossible for the whole network.
    """
    if cfg.finetune_mode not in {"lora", "full"}:
        raise ValueError(f"finetune_mode must be 'lora' or 'full', got {cfg.finetune_mode!r}")
    if cfg.finetune_mode == "lora":
        return cfg

    if cfg.load_in_4bit:
        raise RuntimeError(
            "finetune_mode='full' cannot train 4-bit weights - quantised parameters "
            "have no gradients. Use finetune_mode='lora' (QLoRA) on this GPU, or set "
            "CFG.load_in_4bit = False and a model small enough to fit in bf16."
        )
    cfg.lr = cfg.full_lr
    cfg.epochs = cfg.full_epochs
    cfg.optim = "paged_adamw_8bit"
    cfg.ema_decay = 0.0
    cfg.warmup_ratio = max(cfg.warmup_ratio, 0.03)
    cfg.per_device_batch_size = max(1, cfg.per_device_batch_size // 4)
    cfg.grad_accum = max(1, cfg.grad_accum * 4)

    # bf16 weights + bf16 grads + 8-bit Adam moments, plus activations. The 1.25
    # factor is the measured overhead of gradient checkpointing on this workload.
    billions = params_billions(cfg.model_id)
    need = billions * (2 + 2 + 2) * 1.25
    print(f"[cfg] full fine-tune: ~{need:.0f} GB needed for {billions:g}B params "
          f"vs {vram_gb:.0f} GB available")
    if need > vram_gb:
        print("!! This will almost certainly OOM. Either switch to "
              "finetune_mode='lora', move to a smaller backbone "
              "(Qwen/Qwen3-VL-4B-Instruct or -2B-Instruct), or add GPUs.")
    return cfg


# Order matters: autoscale sets hardware-appropriate defaults, explicit overrides
# then win over them, and the mode/quantisation derivations run last so they see
# the settings that will actually be used.
CFG = autoscale(CFG, GPU_VRAM_GB)
if not CFG.hf_repo_id:
    CFG.hf_repo_id = os.environ.get("HF_REPO_ID", "").strip()

CFG_OVERRIDES = globals().get("CLI_OVERRIDES", {})   # populated by run_pipeline.py
for _key, _val in CFG_OVERRIDES.items():
    if not hasattr(CFG, _key):
        raise SystemExit(f"unknown config field: {_key}")
    setattr(CFG, _key, _val)
if CFG_OVERRIDES:
    print(f"[cfg] overrides from the command line: {CFG_OVERRIDES}")

CFG.patch_grid = infer_patch_grid(CFG.model_id)
if CFG.load_in_4bit:
    CFG.optim = "paged_adamw_8bit"   # keeps optimiser state off the critical path
CFG = apply_finetune_mode(CFG, GPU_VRAM_GB)

# Multi-GPU: HF Trainer shards the batch across devices, so keep the effective
# batch constant rather than multiplying it.
if N_GPU > 1:
    CFG.grad_accum = max(1, CFG.grad_accum // N_GPU)
    print(f"[cfg] {N_GPU} GPUs detected -> grad_accum reduced to {CFG.grad_accum}")

WORK = Path(CFG.work_dir).resolve()
OUT = WORK / CFG.out_dir
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "checkpoints").mkdir(exist_ok=True)
(OUT / "preds").mkdir(exist_ok=True)
(OUT / "cache").mkdir(exist_ok=True)

EFFECTIVE_BATCH = CFG.per_device_batch_size * CFG.grad_accum * max(1, N_GPU)
print(json.dumps({k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(CFG).items()}, indent=2))
print(f"\nfine-tune mode       {CFG.finetune_mode}"
      f"{' (4-bit QLoRA)' if CFG.load_in_4bit else ''}")
print(f"effective batch size = {EFFECTIVE_BATCH}")
print(f"peak lr {CFG.lr:g} -> floor {CFG.lr*CFG.min_lr_ratio:g} over {CFG.epochs:g} epochs")

# ### 2.1 Licence guard
#
# The organisers rejected a model purely on licence grounds, so this is enforced in
# code rather than left to memory. The guard reads the licence from the Hugging Face
# model card and refuses anything outside the commercial-use allow-list. If the hub
# is unreachable, it falls back to a hard-coded table of models whose licences were
# verified while writing this notebook.

PERMISSIVE_LICENCES = {
    "apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause",
    "cc0-1.0", "cc-by-4.0", "unlicense", "isc", "artistic-2.0",
}

# Verified 2026-08. Anything marked False must not be used in this challenge.
KNOWN_LICENCES = {
    "Qwen/Qwen3-VL-8B-Instruct": ("apache-2.0", True),
    "Qwen/Qwen3-VL-4B-Instruct": ("apache-2.0", True),
    "Qwen/Qwen3-VL-2B-Instruct": ("apache-2.0", True),
    "Qwen/Qwen2.5-VL-7B-Instruct": ("apache-2.0", True),
    "Qwen/Qwen2.5-VL-32B-Instruct": ("apache-2.0", True),
    "Qwen/Qwen2-VL-7B-Instruct": ("apache-2.0", True),
    "Qwen/Qwen2-VL-2B-Instruct": ("apache-2.0", True),
    "microsoft/trocr-large-handwritten": ("mit", True),
    "microsoft/trocr-base-handwritten": ("mit", True),
    "microsoft/Florence-2-large": ("mit", True),
    "stepfun-ai/GOT-OCR2_0": ("apache-2.0", True),
    "allenai/olmOCR-7B-0225-preview": ("apache-2.0", True),
    # kraken PP-OCRv6 ports (CTC line recognisers, Section 14.1). Verified 2026-09.
    "small-models-for-glam/kraken-ppocrv6-medium": ("apache-2.0", True),
    "small-models-for-glam/kraken-ppocrv6-small": ("apache-2.0", True),
    "small-models-for-glam/kraken-ppocrv6-tiny": ("apache-2.0", True),
    # --- blocked ---
    "Qwen/Qwen2.5-VL-3B-Instruct": ("qwen-research", False),
    "stanford-oval/churro-3B": ("qwen-research", False),
    "nanonets/Nanonets-OCR-s": ("qwen-research", False),
}


def check_licence(model_id: str, strict: bool = True) -> str:
    """Return the licence id for `model_id`, raising if it is not commercial-friendly."""
    licence, verdict = KNOWN_LICENCES.get(model_id, (None, None))
    source = "static table"
    try:
        from huggingface_hub import model_info

        info = model_info(model_id)
        card = info.cardData or {}
        hub_licence = (card.get("license") or card.get("license_name") or "")
        if isinstance(hub_licence, list):
            hub_licence = hub_licence[0] if hub_licence else ""
        hub_licence = str(hub_licence).strip().lower()
        if hub_licence:
            licence, source = hub_licence, "huggingface hub"
            verdict = hub_licence in PERMISSIVE_LICENCES
    except Exception as exc:  # offline pod, gated repo, rate limit
        print(f"[licence] hub lookup failed for {model_id} ({type(exc).__name__}); using {source}")

    if licence is None:
        msg = (f"[licence] UNKNOWN licence for {model_id}. Verify manually on the model "
               f"card and add it to KNOWN_LICENCES before training.")
        if strict:
            raise RuntimeError(msg)
        print(msg)
        return "unknown"

    flag = "OK" if verdict else "BLOCKED"
    print(f"[licence] {model_id}: {licence} -> {flag} (via {source})")
    if not verdict:
        raise RuntimeError(
            f"{model_id} is licensed '{licence}', which does not permit commercial use. "
            "The organisers explicitly disallow such models in this challenge "
            "(see the churro-3B ruling). Pick an Apache-2.0 / MIT backbone."
        )
    return licence


CFG_LICENCE = check_licence(CFG.model_id)
CTC_LICENCE = check_licence(CFG.ctc_model_id) if CFG.ctc_branch else None

# ==========================================================================
# p02a_repro.py
# ==========================================================================

# ### 2.2 Reproducibility contract
#
# "Re-run and get the same number" needs three separate things, and it is worth
# being precise about which one each mechanism buys:
#
# 1. **Same sampling.** Seeding `random`, `numpy` and `torch` fixes the fold split,
#    the augmentation draws and the StackMix composition. This notebook goes further:
#    every augmented sample is drawn from `default_rng((seed, epoch, index))` rather
#    than from a stream, so sample *i* of epoch *e* is identical no matter how many
#    dataloader workers there are or in what order they happen to run.
# 2. **Same arithmetic.** GPU reductions are non-deterministic by default because
#    floating-point addition is not associative and the kernels race. `deterministic`
#    mode pins cuBLAS workspaces and forbids the non-deterministic kernels, which
#    costs roughly 5-10% throughput and is worth paying for a competition run.
# 3. **Same inputs and same code.** A seed is useless if the CSV or the library
#    versions moved underneath it. `RUN_FINGERPRINT` hashes the actual input files
#    and records the versions, and it travels with every checkpoint pushed to the hub.
#
# What still cannot be promised: a different GPU architecture, or a different CUDA
# or cuBLAS build, can produce a slightly different result from bit-identical inputs.
# The fingerprint records the device so that case is at least visible rather than
# mysterious.

import hashlib
import platform


def set_seed(seed: int, deterministic: bool | None = None) -> None:
    """Seed every RNG this pipeline touches, optionally pinning kernel choice too."""
    if deterministic is None:
        deterministic = getattr(CFG, "deterministic", False)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Must be set before the first cuBLAS handle is created to take effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:
            print(f"[repro] deterministic algorithms unavailable ({exc})")
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    if not Path(path).exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def package_versions() -> dict:
    import importlib

    out = {}
    for name in ("torch", "transformers", "peft", "trl", "accelerate", "datasets",
                 "bitsandbytes", "numpy", "pandas", "cv2", "rapidfuzz", "sklearn"):
        try:
            out[name] = str(getattr(importlib.import_module(name), "__version__", "?"))
        except Exception:
            out[name] = "absent"
    return out


def build_fingerprint(cfg: Config) -> dict:
    """Everything that has to match for two runs to be comparable."""
    inputs = {name: file_digest(WORK / name)
              for name in (cfg.train_csv, cfg.test_csv, cfg.sample_sub_csv)}
    # Config fields that change the numbers, not the plumbing. Hub and I/O settings
    # are excluded so that pushing to a different repo does not invalidate a match.
    volatile = {"hf_repo_id", "hf_private", "hf_push_checkpoints", "hf_push_submission",
                "hf_resume", "hf_token_env", "work_dir", "data_dir", "out_dir",
                "dataloader_workers", "infer_batch_size",
                "ctc_venv", "ctc_torch_index", "ctc_infer_batch"}
    material = {k: v for k, v in asdict(cfg).items() if k not in volatile}
    config_hash = hashlib.sha256(
        json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()[:16]
    fp = {
        "config_hash": config_hash,
        "input_digests": inputs,
        "packages": package_versions(),
        "device": GPU_NAME,
        "n_gpu": N_GPU,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "deterministic": bool(cfg.deterministic),
        "seed": cfg.seed,
    }
    fp["run_id"] = hashlib.sha256(
        json.dumps({k: fp[k] for k in ("config_hash", "input_digests", "seed")},
                   sort_keys=True).encode()).hexdigest()[:12]
    return fp


set_seed(CFG.seed, CFG.deterministic)
RUN_FINGERPRINT = build_fingerprint(CFG)
with open(OUT / "run_fingerprint.json", "w") as fh:
    json.dump(RUN_FINGERPRINT, fh, indent=2)

print(f"[repro] run id        {RUN_FINGERPRINT['run_id']}")
print(f"[repro] config hash   {RUN_FINGERPRINT['config_hash']}")
print(f"[repro] input digests {RUN_FINGERPRINT['input_digests']}")
print(f"[repro] determinism   {'pinned' if CFG.deterministic else 'off (faster, not bit-exact)'}")
print("[repro] versions      " + ", ".join(
    f"{k} {v}" for k, v in RUN_FINGERPRINT["packages"].items() if v != "absent"))

# ==========================================================================
# p02b_hub.py
# ==========================================================================

# ### 2.3 Hugging Face Hub sync
#
# A competition run is worth nothing if the pod is destroyed before the weights are
# safe. Set `CFG.hf_repo_id` (and export a write token as `$HF_TOKEN`) and every
# artefact needed to reproduce the leaderboard score is mirrored to a private repo
# as it is produced, not in one push at the end:
#
# | Uploaded | When | Why it is needed to replay the run |
# |---|---|---|
# | `checkpoints/<tag>/` | as each fold finishes | the adapter (or full model) that produced the score |
# | `val_history_<tag>.csv` | as each fold finishes | proves which step was selected and why |
# | `run_fingerprint.json` | at start-up | input digests + library versions the score is conditional on |
# | `post_config.json` | after tuning | the fusion weights, which move the score by more than a training epoch |
# | `submission.csv`, `run_manifest.json` | at the end | the exact rows that were scored |
#
# Going the other way, `CFG.hf_resume = True` pulls the checkpoints back down and
# skips training entirely, which turns a four-hour run into a decoding-only pass —
# useful for re-tuning post-processing without paying for the GPU hours again.
#
# Nothing here is required: with `hf_repo_id` empty every call is a no-op and the
# notebook behaves exactly as before.

class HubSync:
    """Thin, failure-tolerant wrapper around huggingface_hub uploads.

    Deliberately never raises: a hub outage or an expired token must not destroy a
    training run that has already cost GPU hours. Failures print and continue.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.repo_id = (cfg.hf_repo_id or "").strip()
        self.api = None
        self.enabled = False
        self.errors: list[str] = []
        if not self.repo_id:
            print("[hub] disabled (CFG.hf_repo_id is empty)")
            return
        token = os.environ.get(cfg.hf_token_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        try:
            from huggingface_hub import HfApi

            self.api = HfApi(token=token)
            who = self.api.whoami()["name"]
            self.api.create_repo(self.repo_id, private=cfg.hf_private,
                                 exist_ok=True, repo_type="model")
            self.enabled = True
            print(f"[hub] {who} -> {self.repo_id} "
                  f"({'private' if cfg.hf_private else 'public'})")
        except Exception as exc:
            print(f"[hub] disabled: {type(exc).__name__}: {exc}")
            print(f"[hub] set a write token with  os.environ['{cfg.hf_token_env}'] = '...'")

    def _guard(self, what: str, fn, *args, **kwargs):
        if not self.enabled:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = f"{what}: {type(exc).__name__}: {exc}"
            self.errors.append(msg)
            print(f"[hub] FAILED {msg}")
            return None

    def push_folder(self, local: Path, path_in_repo: str, message: str = "") -> bool:
        local = Path(local)
        if not self.enabled or not local.exists():
            return False
        size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file()) / 1e6
        print(f"[hub] uploading {path_in_repo} ({size:.0f} MB) ...")
        return self._guard(f"upload {path_in_repo}", self.api.upload_folder,
                           folder_path=str(local), path_in_repo=path_in_repo,
                           repo_id=self.repo_id, repo_type="model",
                           commit_message=message or f"add {path_in_repo}") is not None

    def push_file(self, local: Path, path_in_repo: str | None = None,
                  message: str = "") -> bool:
        local = Path(local)
        if not self.enabled or not local.exists():
            return False
        return self._guard(f"upload {local.name}", self.api.upload_file,
                           path_or_fileobj=str(local),
                           path_in_repo=path_in_repo or local.name,
                           repo_id=self.repo_id, repo_type="model",
                           commit_message=message or f"add {local.name}") is not None

    def pull(self, patterns: list[str], dest: Path) -> Path | None:
        """Download matching files from the repo into `dest`. Returns the local root."""
        if not self.enabled:
            return None
        from huggingface_hub import snapshot_download

        return self._guard("snapshot_download", snapshot_download,
                           repo_id=self.repo_id, repo_type="model",
                           allow_patterns=patterns, local_dir=str(dest))

    def write_card(self, body: str) -> None:
        if not self.enabled:
            return
        card = OUT / "README_hub.md"
        card.write_text(body, encoding="utf-8")
        self.push_file(card, "README.md", "update model card")


def model_card(cfg: Config, licence: str, fingerprint: dict,
               runs: list | None = None, scores: dict | None = None) -> str:
    """Model card carrying the numbers and the exact recipe that produced them."""
    base = cfg.model_id
    rows = ""
    for r in runs or []:
        best = r.get("best", {})
        rows += (f"| `{r.get('tag')}` | {best.get('score', float('nan')):.5f} | "
                 f"{best.get('source', '?')} | {best.get('step', '?')} |\n")
    score_lines = "".join(f"- **{k}**: {v}\n" for k, v in (scores or {}).items())
    return f"""---
license: {licence}
base_model: {base}
tags:
  - handwritten-text-recognition
  - ocr
  - historical-documents
  - {"lora" if cfg.finetune_mode == "lora" else "full-finetune"}
library_name: peft
---

# R.O.A.D. Barbados Historic Handwriting - {cfg.finetune_mode.upper()} fine-tune

{"LoRA adapters" if cfg.finetune_mode == "lora" else "Full fine-tune"} of
[`{base}`]({f"https://huggingface.co/{base}"}) for transcribing 17th-19th century
Barbadian deed, will and inventory line crops.

## Results

{score_lines or "_scores are written here when the run completes_"}

| fold | val score | selected from | step |
|---|---|---|---|
{rows or "| _pending_ | | | |"}

## Reproducing this exactly

Run `ROAD_Barbados_HTR_SOTA.ipynb` with the configuration below. The run is
deterministic given the same inputs and device: sampling is seeded per
`(seed, epoch, index)` and cuDNN/cuBLAS determinism is
{"enabled" if cfg.deterministic else "**disabled** for this run"}.

- run id: `{fingerprint.get('run_id')}`
- config hash: `{fingerprint.get('config_hash')}`
- input digests: `{fingerprint.get('input_digests')}`
- seed: `{cfg.seed}`
- device: `{fingerprint.get('device')}` x{fingerprint.get('n_gpu')}, CUDA `{fingerprint.get('cuda')}`
- key versions: `{ {k: v for k, v in fingerprint.get('packages', {}).items() if k in ('torch', 'transformers', 'peft', 'trl')} }`

```python
CFG.finetune_mode = "{cfg.finetune_mode}"
CFG.model_id      = "{base}"
CFG.seed          = {cfg.seed}
CFG.epochs        = {cfg.epochs}
CFG.lr            = {cfg.lr:g}
CFG.lora_r        = {cfg.lora_r}
CFG.lora_alpha    = {cfg.lora_alpha}
CFG.line_height   = {cfg.line_height}
CFG.stackmix      = {cfg.stackmix}
CFG.run_grpo      = {cfg.run_grpo}
```

## Data and licence

Trained only on the challenge's own `Train.csv` images, which are licensed for use
within the competition and are **not** redistributed here. This repository contains
weights and run metadata only.
"""


HUB = HubSync(CFG)
HUB.push_file(OUT / "run_fingerprint.json", "run_fingerprint.json", "record run fingerprint")

# ==========================================================================
# p02c_notify.py
# ==========================================================================

# ### 2.4 Email notifications
#
# A five-fold run with GRPO is several hours on a rented pod, and the failure modes
# that matter are the silent ones: an OOM at fold three, an expired hub token, a
# reward function that quietly returns the same value for every completion. This
# sends mail at the points where a human would actually want to intervene:
#
# | Event | Sent when | Why you care |
# |---|---|---|
# | `run started` | after the config is resolved | confirms the pod really is training what you think |
# | `checkpoint uploaded` | each fold, after the hub push | the weights are safe; includes the fold's validation score |
# | `grpo verdict` | each GRPO stage | accepted or rejected, with the delta |
# | `run finished` | after the submission is written | final out-of-fold score and where the file is |
# | `run FAILED` | any uncaught exception | full traceback, so you can fix it without opening the pod |
#
# Credentials come from the environment (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`,
# `SMTP_PASSWORD`, `NOTIFY_EMAIL_TO`), loaded from a gitignored `.env` by `Section 2`.
# Nothing is hard-coded here, so this notebook can be shared without leaking a
# password. With Gmail the password must be a 16-character **App Password**, not the
# account password.
#
# Like the hub layer, the notifier never raises: a mail server that refuses a
# connection must not kill a training run.

import smtplib
import traceback
from email.message import EmailMessage


class Notifier:
    """Sends short status emails over SMTP, and never lets a mail failure escalate."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.to = os.environ.get("NOTIFY_EMAIL_TO", "").strip()
        self.host = os.environ.get("SMTP_HOST", "").strip()
        self.port = int(os.environ.get("SMTP_PORT", "587") or 587)
        self.user = os.environ.get("SMTP_USER", "").strip()
        self.password = os.environ.get("SMTP_PASSWORD", "").strip()
        self.enabled = bool(cfg.notify_email and self.to and self.host
                            and self.user and self.password)
        self.sent = 0
        self.failures: list[str] = []
        self.t0 = time.time()
        if self.enabled:
            print(f"[notify] email -> {self.to} via {self.host}:{self.port}")
        elif not cfg.notify_email:
            print("[notify] disabled (CFG.notify_email = False)")
        else:
            missing = [n for n, v in (("SMTP_HOST", self.host), ("SMTP_USER", self.user),
                                      ("SMTP_PASSWORD", self.password),
                                      ("NOTIFY_EMAIL_TO", self.to)) if not v]
            print(f"[notify] disabled (missing {', '.join(missing)} - see .env.example)")

    def _elapsed(self) -> str:
        m, s = divmod(int(time.time() - self.t0), 60)
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

    def send(self, subject: str, body: str, urgent: bool = False) -> bool:
        prefix = "[ROAD-HTR]" + (" FAILURE" if urgent else "")
        line = f"{prefix} {subject}"
        print(f"[notify] {line}")
        if not self.enabled:
            return False
        run = RUN_FINGERPRINT.get("run_id", "?")
        footer = (f"\n\n--\nrun {run} | {GPU_NAME} | elapsed {self._elapsed()}"
                  f"\nmodel {self.cfg.model_id} | mode {self.cfg.finetune_mode}"
                  + (f"\nhub https://huggingface.co/{self.cfg.hf_repo_id}"
                     if self.cfg.hf_repo_id else ""))
        msg = EmailMessage()
        msg["Subject"] = line
        msg["From"] = self.user
        msg["To"] = self.to
        msg.set_content(body + footer)
        try:
            with smtplib.SMTP(self.host, self.port, timeout=30) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.send_message(msg)
            self.sent += 1
            return True
        except Exception as exc:
            note = f"{type(exc).__name__}: {exc}"
            self.failures.append(note)
            print(f"[notify] send failed ({note})")
            return False

    # ------------------------------------------------------------- events --
    def run_started(self) -> None:
        self.send(
            f"run started on {GPU_NAME}",
            f"Training {self.cfg.model_id} ({self.cfg.finetune_mode}) on folds "
            f"{list(self.cfg.train_folds)}.\n"
            f"epochs {self.cfg.epochs:g} | lr {self.cfg.lr:g} | batch {EFFECTIVE_BATCH}\n"
            f"stackmix {self.cfg.stackmix} | grpo {self.cfg.run_grpo} | "
            f"deterministic {self.cfg.deterministic}\n"
            f"run id {RUN_FINGERPRINT['run_id']} (config {RUN_FINGERPRINT['config_hash']})",
        )

    def checkpoint(self, tag: str, best: dict, minutes: float, pushed: bool) -> None:
        where = (f"https://huggingface.co/{self.cfg.hf_repo_id}/tree/main/checkpoints/{tag}"
                 if pushed else f"local only: {best.get('path')}")
        self.send(
            f"checkpoint {tag} - val {best.get('score', float('nan')):.5f}",
            f"Fold finished in {minutes:.1f} min.\n\n"
            f"best score  {best.get('score', float('nan')):.5f}\n"
            f"selected    {best.get('source')} at step {best.get('step')}\n"
            f"uploaded    {where}",
        )

    def grpo(self, fold: int, scores: dict, accepted: bool) -> None:
        self.send(
            f"grpo fold {fold} {'accepted' if accepted else 'rejected'} "
            f"({scores['delta']:+.5f})",
            f"SFT  {scores['sft']:.5f}\nGRPO {scores['grpo']:.5f}\n"
            f"delta {scores['delta']:+.5f}\n\n"
            + ("Polished weights will be used for decoding."
               if accepted else "Polished weights discarded; the SFT adapter continues."),
        )

    def finished(self, oof: float, baseline: float, path: Path) -> None:
        self.send(
            f"run finished - OOF {oof:.5f}",
            f"out-of-fold score  {oof:.5f}\n"
            f"before fusion      {baseline:.5f}  ({oof - baseline:+.5f} from post-processing)\n"
            f"submission         {path}",
        )

    def failed(self, where: str, detail: str) -> None:
        self.send(f"{where}", f"The run stopped.\n\n{detail}"[:8000], urgent=True)

    def install_crash_handler(self) -> None:
        """Email any uncaught exception, then print it as usual.

        Replaces the bootstrap handler that `run_pipeline.py` installs before this
        object exists, which is why it chains to the interpreter default rather
        than to whatever hook is currently registered - otherwise a crash would be
        reported twice.
        """
        def hook(exc_type, exc, tb):
            self.failed("run crashed",
                        "".join(traceback.format_exception(exc_type, exc, tb)))
            sys.__excepthook__(exc_type, exc, tb)

        sys.excepthook = hook


NOTIFY = Notifier(CFG)

# --- installed by build_script.py: email any uncaught exception ---------------
NOTIFY.install_crash_handler()

# ==========================================================================
# p02d_tb.py
# ==========================================================================

# ### 2.5 TensorBoard
#
# The Trainer logs loss and learning rate on its own, but loss is not the quantity
# being optimised here — the leaderboard scores edit distance, and the two diverge
# late in training exactly when it matters. So the same writer also receives the
# **generation-based** metrics: every validation pass logs `val/score`, `val/cer`
# and `val/wer` for both the live weights and the EMA, and GRPO logs its reward and
# the accept/reject delta. That makes the curves answer the questions you actually
# ask mid-run: is the cosine tail still buying anything, is EMA ahead of raw, did
# GRPO help.
#
# Logs are written per fold to `artifacts/runs/<tag>/`, which is both the local
# TensorBoard layout and the layout the Hugging Face Hub expects:
#
# ```bash
# tensorboard --logdir artifacts/runs --port 6006     # local / port-forwarded
# ```
#
# When `hf_repo_id` is set, `runs/` is pushed with each fold and the repo grows a
# **Training metrics** tab that renders the same curves in the browser, with no pod
# access and no port forwarding — which is the point when the job is running
# headless on a machine you would rather not SSH into.

class TBLogger:
    """Per-run scalar writer, shared between the Trainer and our own callbacks.

    Wraps SummaryWriter so that a missing tensorboard install degrades to a no-op
    instead of taking down a training run over a logging dependency.
    """

    root = OUT / "runs"

    def __init__(self, tag: str, enabled: bool = True):
        self.tag = tag
        self.dir = self.root / tag
        self.writer = None
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.dir))
        except Exception as exc:
            print(f"[tb] disabled ({type(exc).__name__}: {exc}) - pip install tensorboard")

    def scalar(self, name: str, value, step: int) -> None:
        if self.writer is None or value is None:
            return
        try:
            self.writer.add_scalar(name, float(value), int(step))
        except Exception:
            pass

    def scalars(self, values: dict, step: int, prefix: str = "") -> None:
        for key, val in values.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                self.scalar(f"{prefix}{key}", val, step)

    def text(self, name: str, body: str, step: int = 0) -> None:
        if self.writer is None:
            return
        try:
            self.writer.add_text(name, body, int(step))
        except Exception:
            pass

    def hparams(self, cfg: Config, metrics: dict) -> None:
        """One row per run in TensorBoard's HPARAMS tab, for comparing experiments."""
        if self.writer is None:
            return
        keep = ("finetune_mode", "model_id", "seed", "epochs", "lr", "lora_r",
                "lora_alpha", "line_height", "stackmix", "stackmix_ratio",
                "neftune_alpha", "ema_decay", "run_grpo")
        hp = {k: (str(v) if isinstance(v, (str, bool)) else v)
              for k, v in asdict(cfg).items() if k in keep}
        try:
            self.writer.add_hparams(hp, {f"hparam/{k}": float(v) for k, v in metrics.items()})
        except Exception as exc:
            print(f"[tb] hparams skipped ({type(exc).__name__})")

    def flush(self) -> None:
        if self.writer is not None:
            try:
                self.writer.flush()
            except Exception:
                pass

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception:
                pass
            self.writer = None


def push_tensorboard(tag: str | None = None, message: str = "update metrics") -> None:
    """Mirror the event files so the hub renders its Training metrics tab."""
    if not (CFG.tensorboard and HUB.enabled):
        return
    local = TBLogger.root / tag if tag else TBLogger.root
    HUB.push_folder(local, f"runs/{tag}" if tag else "runs", message)


TBLogger.root.mkdir(parents=True, exist_ok=True)
print(f"[tb] {'enabled' if CFG.tensorboard else 'disabled'} -> {TBLogger.root}")
print(f"[tb] view locally with:  tensorboard --logdir {TBLogger.root} --port 6006")
if CFG.hf_repo_id:
    print(f"[tb] and in the browser at "
          f"https://huggingface.co/{CFG.hf_repo_id}/tensorboard")

# ==========================================================================
# p02e_progress.py
# ==========================================================================

# ### 2.6 Progress reporting
#
# A full run is a few hours across a dozen stages, and half of them are silent loops
# over thousands of images. Two things make that legible, and both have to work in a
# terminal *and* in a log file that nobody is watching live:
#
# * **`progress(...)`** wraps `tqdm` with one house style. In a terminal it behaves
#   normally. When stdout is not a TTY — which is every background run — it switches
#   to ASCII and refreshes at most every `PROGRESS_INTERVAL` seconds (30 by default)
#   instead of ten times a second. That matters more than it sounds: `tqdm` redraws
#   with a carriage return, so a default-rate bar writes tens of thousands of updates
#   into a log file. At one line every 30 seconds the same bar becomes a readable
#   timeline of the run.
# * **`stage(...)`** prints a banner and records how long each phase took. The
#   summary at the end tells you where the wall clock actually went, which is the
#   information you need to decide what to cut when the deadline is close.
#
# `PROGRESS_INTERVAL` is read from the environment so a background run can be made
# chattier (`PROGRESS_INTERVAL=5`) without editing anything.

from tqdm.auto import tqdm

INTERACTIVE = bool(getattr(sys.stdout, "isatty", lambda: False)())
PROGRESS_INTERVAL = float(os.environ.get("PROGRESS_INTERVAL", "30"))
RUN_T0 = time.time()


def fmt_hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def progress(iterable=None, *, total=None, desc="", unit="it", leave=True, **kw):
    """tqdm with one house style, tuned for whichever output we actually have."""
    opts = dict(desc=desc, unit=unit, leave=leave, total=total,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]{postfix}")
    if INTERACTIVE:
        opts.update(ncols=100, mininterval=0.2, smoothing=0.3)
    else:
        # One tidy, timestamped line every PROGRESS_INTERVAL seconds.
        opts.update(ncols=80, ascii=True, mininterval=PROGRESS_INTERVAL,
                    miniters=1, smoothing=0.1, file=sys.stdout)
    opts.update(kw)
    return tqdm(iterable, **opts)


STAGES: list[dict] = []


def stage(name: str) -> None:
    """Close the previous phase, open a new one, and say where we are."""
    now = time.time()
    if STAGES:
        STAGES[-1]["seconds"] = now - STAGES[-1]["t0"]
    STAGES.append({"name": name, "t0": now, "seconds": None})
    print(f"\n{'=' * 78}")
    print(f"[stage {len(STAGES)}] {name}    (+{fmt_hms(now - RUN_T0)} into the run)")
    print("=" * 78)


def stage_summary() -> "pd.DataFrame":
    now = time.time()
    if STAGES and STAGES[-1]["seconds"] is None:
        STAGES[-1]["seconds"] = now - STAGES[-1]["t0"]
    total = max(now - RUN_T0, 1e-6)
    df = pd.DataFrame([{"stage": s["name"],
                        "minutes": round((s["seconds"] or 0) / 60, 1),
                        "share": f"{100 * (s['seconds'] or 0) / total:5.1f}%"}
                       for s in STAGES])
    print(f"\nwall clock {fmt_hms(total)} over {len(STAGES)} stages")
    print(df.to_string(index=False))
    return df


print(f"[progress] {'interactive terminal' if INTERACTIVE else 'headless'}: "
      f"bars refresh every {0.2 if INTERACTIVE else PROGRESS_INTERVAL:g}s")

# ==========================================================================
# p03_data.py
# ==========================================================================

# ## 3. Data
#
# `images.zip` (≈421 MB, 5 472 crops) is fetched from the URL on the challenge Data
# tab and unpacked once. The CSVs are expected next to the notebook. Every ID is
# resolved through a single filename index so a nested or renamed extraction
# directory does not break anything.

DATA = WORK / CFG.data_dir
DATA.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[data] {dest.name} already present ({dest.stat().st_size/1e6:.0f} MB)")
        return dest
    import urllib.request

    print(f"[data] downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0)) or None
        bar = progress(total=total, desc="download", unit="B",
                       unit_scale=True, unit_divisor=1024, bar_format=None)
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)
            bar.update(len(buf))
        bar.close()
    tmp.rename(dest)
    return dest


def ensure_images() -> Path:
    zip_path = DATA / "images.zip"
    img_dir = DATA / CFG.image_subdir
    if not (img_dir.exists() and any(img_dir.rglob("*.jpg"))):
        download(CFG.images_url, zip_path)
        print("[data] extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA)
    # The archive may unpack to data/images or data/<something>/images
    candidates = [p for p in [img_dir, DATA] if p.exists()]
    for root in candidates:
        jpgs = list(root.rglob("*.jpg"))
        if jpgs:
            return jpgs[0].parent
    raise FileNotFoundError("No .jpg files found after extraction")


stage("data: download and index")
IMAGE_DIR = ensure_images()
IMAGE_INDEX = {p.stem: p for p in IMAGE_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}}
print(f"[data] image dir  {IMAGE_DIR}")
print(f"[data] indexed    {len(IMAGE_INDEX)} images")


def image_path(image_id: str) -> Path | None:
    return IMAGE_INDEX.get(str(image_id).strip())


train_df = pd.read_csv(WORK / CFG.train_csv)
test_df = pd.read_csv(WORK / CFG.test_csv)
sample_sub = pd.read_csv(WORK / CFG.sample_sub_csv)

train_df["ID"] = train_df["ID"].astype(str).str.strip()
test_df["ID"] = test_df["ID"].astype(str).str.strip()

missing_train = [i for i in train_df.ID if image_path(i) is None]
missing_test = [i for i in test_df.ID if image_path(i) is None]
print(f"[data] train rows {len(train_df)} | missing images {len(missing_train)}")
print(f"[data] test rows  {len(test_df)} | missing images {len(missing_test)}")
assert not missing_test, f"Test images missing: {missing_test[:5]}"
if missing_train:
    train_df = train_df[~train_df.ID.isin(missing_train)].reset_index(drop=True)

# ### 3.1 Label handling
#
# The transcriptions use archival conventions that must survive round-tripping:
# `^` marks a superscript (`y^e`, `15^th`, `w^tsoever`), `~` marks a scribal
# abbreviation stroke, and original spelling is preserved (`publique`, `prsence`).
# Normalisation is therefore deliberately minimal — Unicode look-alikes are folded
# onto the ASCII characters that actually occur in the training labels, whitespace is
# collapsed, and nothing else is touched.

import unicodedata

UNICODE_FOLD = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u2009": " ", "\u200b": "",
    "\u2026": "...", "\ufeff": "",
}
_FOLD_RE = re.compile("|".join(map(re.escape, UNICODE_FOLD)))


def normalize_text(s: Any) -> str:
    """Minimal, loss-free-for-this-corpus label normalisation."""
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = _FOLD_RE.sub(lambda m: UNICODE_FOLD[m.group()], s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(s.split()).strip()


train_df["Target"] = train_df["Target"].map(normalize_text)
train_df = train_df[train_df["Target"].str.len() > 0].reset_index(drop=True)

CHARSET = sorted(Counter("".join(train_df["Target"])).keys())
CHAR_FREQ = Counter("".join(train_df["Target"]))
ALLOWED = set(CHARSET)
print(f"[labels] {len(train_df)} usable rows | {len(CHARSET)} distinct characters")
print("[labels] charset:", "".join(CHARSET))

# Characters seen fewer than 3 times are almost certainly annotation noise; a
# prediction containing them is more likely a hallucination than a real glyph.
RARE_CHARS = {c for c, n in CHAR_FREQ.items() if n < 3}
print("[labels] rare chars (<3 occurrences):", sorted(RARE_CHARS))

WORD_FREQ = Counter(w for t in train_df["Target"] for w in t.split())
VOCAB = set(WORD_FREQ)
print(f"[labels] vocabulary {len(VOCAB)} types / {sum(WORD_FREQ.values())} tokens")
print("[labels] most common:", WORD_FREQ.most_common(15))


def restrict_charset(s: str) -> str:
    """Drop characters the model can only have hallucinated."""
    return "".join(ch for ch in s if ch in ALLOWED)

# ### 3.2 Exploratory analysis
#
# Two facts from this cell drive the whole image pipeline:
#
# * **Aspect ratios are extreme** (median ≈ 16.6:1, p90 ≈ 23.7:1). A naive square
#   resize destroys the glyphs, so `Section 6` normalises height and lets width run.
# * **Crops include slivers of the lines above and below.** The label covers only the
#   centre line, so vertical ink-cropping is disabled and the prompt tells the model
#   explicitly to transcribe the central line only.

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def image_stats(ids: Sequence[str], limit: int = 1500) -> pd.DataFrame:
    rows = []
    for i in list(ids)[:limit]:
        p = image_path(i)
        if p is None:
            continue
        with Image.open(p) as im:
            w, h = im.size
        rows.append({"ID": i, "w": w, "h": h, "aspect": w / max(h, 1)})
    return pd.DataFrame(rows)


tr_stats = image_stats(train_df.ID)
te_stats = image_stats(test_df.ID)
tr_stats = tr_stats.merge(train_df, on="ID", how="left")
tr_stats["n_chars"] = tr_stats["Target"].str.len()
tr_stats["n_words"] = tr_stats["Target"].str.split().apply(len)
tr_stats["px_per_char"] = tr_stats["w"] / tr_stats["n_chars"].clip(lower=1)

summary = pd.DataFrame({
    "train": tr_stats[["w", "h", "aspect", "n_chars", "n_words", "px_per_char"]].median(),
    "train_p90": tr_stats[["w", "h", "aspect", "n_chars", "n_words", "px_per_char"]].quantile(0.9),
})
print(summary.round(2))
print("\ntest image geometry vs train (median / p90):")
print(pd.DataFrame({
    "test_med": te_stats[["w", "h", "aspect"]].median(),
    "test_p90": te_stats[["w", "h", "aspect"]].quantile(0.9),
    "train_med": tr_stats[["w", "h", "aspect"]].median(),
    "train_p90": tr_stats[["w", "h", "aspect"]].quantile(0.9),
}).round(1))

fig, ax = plt.subplots(2, 3, figsize=(16, 7))
ax[0, 0].hist(tr_stats.aspect, bins=50, color="#3b6ea5")
ax[0, 0].hist(te_stats.aspect, bins=50, color="#c8553d", alpha=0.55)
ax[0, 0].set_title("aspect ratio (blue=train, red=test)")
ax[0, 1].hist(tr_stats.h, bins=50, color="#3b6ea5")
ax[0, 1].set_title("crop height (px)")
ax[0, 2].hist(tr_stats.n_chars, bins=40, color="#3b6ea5")
ax[0, 2].set_title("label length (chars)")
ax[1, 0].hist(tr_stats.n_words, bins=range(0, 26), color="#3b6ea5")
ax[1, 0].set_title("label length (words)")
ax[1, 1].hist(np.log10(tr_stats.px_per_char), bins=40, color="#3b6ea5")
ax[1, 1].set_title("log10 px per character")
top = CHAR_FREQ.most_common(30)
ax[1, 2].bar([c for c, _ in top], [n for _, n in top], color="#3b6ea5")
ax[1, 2].set_yscale("log")
ax[1, 2].set_title("character frequency (top 30)")
plt.tight_layout()
plt.savefig(OUT / "eda_overview.png", dpi=110)
plt.close(fig)
print(f"\n[eda] saved {OUT/'eda_overview.png'}")

# Visual check: what the model actually sees, and why vertical cropping is off.
fig, axes = plt.subplots(6, 1, figsize=(16, 9))
for ax_, (_, row) in zip(axes, train_df.sample(6, random_state=3).iterrows()):
    ax_.imshow(Image.open(image_path(row.ID)).convert("L"), cmap="gray")
    ax_.set_title(row.Target, fontsize=8, loc="left")
    ax_.axis("off")
plt.tight_layout()
plt.savefig(OUT / "eda_samples.png", dpi=110)
plt.close(fig)
print(f"[eda] saved {OUT/'eda_samples.png'} - note the partial lines at top/bottom")

# ==========================================================================
# p04_metric.py
# ==========================================================================

# ## 4. The competition metric
#
# From the Evaluation tab: two error metrics, **weighted WER** and **weighted CER**,
# combined as `0.5 * WER_score + 0.5 * CER_score` where each score is `1 - error`.
# "Longer reference transcriptions are weighted more heavily, so errors on longer
# samples contribute more to the final score than errors on very short samples."
#
# That description matches a length-power weighting, `w_i = len(ref_i) ** f`, with
# the starter pack's default `weight_factor = 0.5`. Because the exact server-side
# formula is not published, three variants are computed:
#
# * `power` — per-sample error rate averaged with weights `len ** f` (**primary**),
# * `micro` — pooled edits over pooled reference length (`f = 1`, the classic
#   corpus-level WER/CER),
# * `macro` — unweighted mean of per-sample error rates.
#
# All three move together; `power` is used for model selection and the others are
# printed as a guard against over-fitting one particular definition. Missing or empty
# predictions are scored as fully incorrect, exactly as the rules state.

from rapidfuzz.distance import Levenshtein

WEIGHT_FACTOR = 0.5


def _tokens(s: str, level: str) -> list:
    return list(s) if level == "char" else s.split()


def error_rates(refs: Sequence[str], hyps: Sequence[str], level: str,
                weight_factor: float = WEIGHT_FACTOR) -> dict:
    """Return power / micro / macro error rates for one level (char or word)."""
    num_p = den_p = num_m = den_m = 0.0
    per_sample = []
    for ref, hyp in zip(refs, hyps):
        r = _tokens(ref, level)
        h = _tokens(hyp if isinstance(hyp, str) else "", level)
        n = max(len(r), 1)
        d = Levenshtein.distance(r, h)
        rate = d / n
        per_sample.append(rate)
        w = n ** weight_factor
        num_p += w * rate
        den_p += w
        num_m += d
        den_m += n
    return {
        "power": num_p / max(den_p, 1e-9),
        "micro": num_m / max(den_m, 1e-9),
        "macro": float(np.mean(per_sample)) if per_sample else 1.0,
        "per_sample": np.asarray(per_sample),
    }


def competition_score(refs: Sequence[str], hyps: Sequence[str],
                      weight_factor: float = WEIGHT_FACTOR,
                      verbose: bool = False) -> dict:
    """Leaderboard-style score: 0.5*(1-WER) + 0.5*(1-CER)."""
    cer = error_rates(refs, hyps, "char", weight_factor)
    wer = error_rates(refs, hyps, "word", weight_factor)
    out = {
        "score": 0.5 * (1 - wer["power"]) + 0.5 * (1 - cer["power"]),
        "cer": cer["power"],
        "wer": wer["power"],
        "cer_micro": cer["micro"],
        "wer_micro": wer["micro"],
        "score_micro": 0.5 * (1 - wer["micro"]) + 0.5 * (1 - cer["micro"]),
        "cer_macro": cer["macro"],
        "wer_macro": wer["macro"],
        "n": len(refs),
    }
    if verbose:
        print(f"score {out['score']:.5f} | CER {out['cer']:.4f} | WER {out['wer']:.4f} "
              f"| (micro: score {out['score_micro']:.5f}, CER {out['cer_micro']:.4f}, "
              f"WER {out['wer_micro']:.4f}) | n={out['n']}")
    return out


def score_frames(gt: pd.DataFrame, pred: pd.DataFrame, id_col: str = "ID",
                 text_col: str = "Target", verbose: bool = True) -> dict:
    """Join on ID the way the grader does: unmatched IDs count as empty predictions."""
    p = pred.set_index(id_col)[text_col].map(normalize_text).to_dict()
    refs = [normalize_text(t) for t in gt[text_col]]
    hyps = [p.get(i, "") for i in gt[id_col]]
    missing = sum(1 for i in gt[id_col] if i not in p)
    if missing:
        print(f"!! {missing} reference IDs have no prediction - scored as empty")
    return competition_score(refs, hyps, verbose=verbose)


# --------------------------------------------------------------------------
# Unit tests: a metric you have not tested is a metric you do not have.
# --------------------------------------------------------------------------
def _test_metric() -> None:
    perfect = competition_score(["the quick brown fox"], ["the quick brown fox"])
    assert abs(perfect["score"] - 1.0) < 1e-9, perfect

    empty = competition_score(["abc def"], [""])
    assert abs(empty["cer"] - 1.0) < 1e-9 and abs(empty["wer"] - 1.0) < 1e-9, empty
    assert abs(empty["score"]) < 1e-9, empty

    one_char = competition_score(["abcd"], ["abcx"])
    assert abs(one_char["cer"] - 0.25) < 1e-9, one_char
    assert abs(one_char["wer"] - 1.0) < 1e-9, one_char

    # Longer references must dominate: the same relative error on a long line moves
    # the weighted score more than on a short one.
    long_ref, short_ref = "a" * 100, "a" * 4
    hit_long = competition_score([long_ref, short_ref], ["b" * 100, short_ref])
    hit_short = competition_score([long_ref, short_ref], [long_ref, "b" * 4])
    assert hit_long["cer"] > hit_short["cer"], (hit_long["cer"], hit_short["cer"])

    # Word-level insertion
    ins = error_rates(["a b c"], ["a b c d"], "word")
    assert abs(ins["power"] - 1 / 3) < 1e-9, ins

    # Frame joining with a missing ID
    gt = pd.DataFrame({"ID": ["x", "y"], "Target": ["hello world", "foo bar"]})
    pr = pd.DataFrame({"ID": ["x"], "Target": ["hello world"]})
    joined = score_frames(gt, pr, verbose=False)
    assert 0.4 < joined["score"] < 0.6, joined
    print("[metric] all unit tests passed")


_test_metric()

# Reference points on this dataset, so a validation number can be sanity-checked:
_probe_refs = train_df["Target"].tolist()[:500]
print("[metric] blank submission        ->", round(competition_score(_probe_refs, [""] * 500)["score"], 4))
print("[metric] lowercased references   ->", round(competition_score(_probe_refs, [t.lower() for t in _probe_refs])["score"], 4))
print("[metric] 1 char in 20 corrupted  ->", round(competition_score(
    _probe_refs,
    ["".join("*" if (i % 20 == 7) else c for i, c in enumerate(t)) for t in _probe_refs],
)["score"], 4))

# ==========================================================================
# p05_split.py
# ==========================================================================

# ## 5. Cross-validation split
#
# The corpus contains repeated legal formulae ("Sealed and delivered in the presence
# of", "To all Christian People to whome these presents shall come") captured from
# different scans. If a near-duplicate line lands in both train and validation the
# local score becomes optimistic and checkpoint selection degrades.
#
# Lines are therefore clustered by text similarity (union-find over pairs with
# normalised Levenshtein similarity ≥ 0.90) and the clusters are kept intact by
# `StratifiedGroupKFold`, stratified on label length so every fold sees the same mix
# of short and long lines.

from rapidfuzz import process as rf_process
from sklearn.model_selection import StratifiedGroupKFold


def build_text_groups(texts: Sequence[str], threshold: float = 0.90) -> np.ndarray:
    """Union-find clustering of near-duplicate transcriptions."""
    keys = [" ".join(re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()) for t in texts]
    n = len(keys)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Exact matches first (cheap), then fuzzy neighbours.
    by_key: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_key[k].append(i)
    for idxs in by_key.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    sim = rf_process.cdist(
        keys, keys, scorer=Levenshtein.normalized_similarity,
        score_cutoff=threshold, workers=-1, dtype=np.float32,
    )
    rows, cols = np.nonzero(sim)
    for i, j in zip(rows, cols):
        if i < j:
            union(int(i), int(j))

    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


t0 = time.time()
train_df["group"] = build_text_groups(train_df["Target"].tolist())
n_groups = train_df["group"].nunique()
clustered = int((train_df.groupby("group").size() > 1).sum())
print(f"[split] {n_groups} text groups from {len(train_df)} lines "
      f"({clustered} groups hold near-duplicates) in {time.time()-t0:.1f}s")

train_df["n_chars"] = train_df["Target"].str.len()
train_df["n_words"] = train_df["Target"].str.split().apply(len)
train_df["len_bin"] = pd.qcut(train_df["n_chars"], q=6, labels=False, duplicates="drop")

sgkf = StratifiedGroupKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.fold_seed)
train_df["fold"] = -1
for fold, (_, val_idx) in enumerate(sgkf.split(train_df, train_df["len_bin"], train_df["group"])):
    train_df.loc[train_df.index[val_idx], "fold"] = fold

assert (train_df["fold"] >= 0).all()
print(train_df.groupby("fold").agg(
    n=("ID", "size"), med_chars=("n_chars", "median"), med_words=("n_words", "median")
))

# Leakage check: no group may appear in two folds.
leaks = train_df.groupby("group")["fold"].nunique()
assert (leaks == 1).all(), f"{int((leaks > 1).sum())} groups leak across folds"
print("[split] no group leaks across folds")

train_df.to_csv(OUT / "train_folds.csv", index=False)


def fold_frames(fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df[train_df.fold != fold].reset_index(drop=True)
    va = train_df[train_df.fold == fold].reset_index(drop=True)
    return tr, va

# ==========================================================================
# p06_images.py
# ==========================================================================

# ## 6. Image pipeline
#
# Line crops are 6:1 to 34:1 and the token budget of a ViT is spent per 32×32 pixel
# cell, so the geometry decisions here matter more than anything else in the model.
#
# 1. **Background flattening** — divide the crop by a heavily median-blurred copy of
#    itself. This removes parchment staining and uneven scanner illumination while
#    leaving ink contrast intact.
# 2. **Deskew** — the rotation in ±3° that maximises the variance of the horizontal
#    ink projection, estimated on a 64px-tall thumbnail so it costs ~1 ms.
# 3. **Horizontal ink crop only** — blank left/right margins are trimmed. Vertical
#    cropping is *not* performed: the crops deliberately contain slivers of the lines
#    above and below, and cutting them away would sometimes remove ascenders and
#    descenders of the target line itself.
# 4. **Height normalisation to the patch grid** — resize to `line_height`, cap width
#    at `max_line_width`, and snap both sides to a multiple of 32 so the processor's
#    own `smart_resize` becomes a no-op and the token count is predictable.
#
# At the defaults a median line becomes 160×2656 → 5×83 = 415 visual tokens, roughly
# 3 tokens per character.

import cv2

cv2.setNumThreads(0)  # dataloader workers already provide the parallelism

CACHE_DIR = OUT / "cache" / f"h{CFG.line_height}_w{CFG.max_line_width}"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def read_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:  # non-ASCII paths on Windows
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def flatten_background(gray: np.ndarray) -> np.ndarray:
    k = max(11, (min(gray.shape[:2]) // 2) | 1)
    k = min(k, 51) | 1
    bg = cv2.medianBlur(gray, k)
    out = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    return np.clip(out, 0, 255).astype(np.uint8)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    m = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 41, 12)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def estimate_skew(gray: np.ndarray, max_deg: float = 3.0, step: float = 0.5) -> float:
    """Angle maximising the variance of the horizontal ink projection."""
    h, w = gray.shape[:2]
    scale = 64.0 / max(h, 1)
    small = cv2.resize(gray, (max(int(w * scale), 32), 64), interpolation=cv2.INTER_AREA)
    mask = (ink_mask(small) > 0).astype(np.float32)
    if mask.sum() < 20:
        return 0.0
    best_angle, best_score = 0.0, -1.0
    centre = (mask.shape[1] / 2, mask.shape[0] / 2)
    for angle in np.arange(-max_deg, max_deg + 1e-6, step):
        if abs(angle) < 1e-6:
            rot = mask
        else:
            M = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
            rot = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return best_angle


def rotate_keep(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-3:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def horizontal_ink_crop(gray: np.ndarray, pad_frac: float = 0.02) -> np.ndarray:
    m = ink_mask(gray)
    col = m.sum(axis=0)
    thr = max(col.max() * 0.02, 1.0)
    on = np.nonzero(col > thr)[0]
    if on.size < 2:
        return gray
    pad = int(gray.shape[0] * pad_frac) + 2
    a = max(int(on[0]) - pad, 0)
    b = min(int(on[-1]) + pad, gray.shape[1])
    if b - a < gray.shape[0]:  # never crop into a sliver
        return gray
    return gray[:, a:b]


def snap(x: float, grid: int, lo: int, hi: int) -> int:
    v = int(round(x / grid)) * grid
    return int(min(max(v, max(lo, grid)), hi))


def target_size(w: int, h: int, cfg: Config, scale: float = 1.0) -> tuple[int, int]:
    """Aspect-preserving size on the patch grid."""
    s = (cfg.line_height * scale) / max(h, 1)
    s = min(s, cfg.max_upscale)
    tw, th = w * s, h * s
    if tw > cfg.max_line_width:
        k = cfg.max_line_width / tw
        tw, th = tw * k, th * k
    th = snap(th, cfg.patch_grid, cfg.min_line_height, cfg.line_height * 2)
    tw = snap(tw, cfg.patch_grid, cfg.min_line_width, cfg.max_line_width)
    return int(tw), int(th)


def normalize_line(gray: np.ndarray, cfg: Config, scale: float = 1.0) -> np.ndarray:
    if cfg.background_flatten:
        gray = flatten_background(gray)
    if cfg.deskew:
        gray = rotate_keep(gray, estimate_skew(gray, cfg.deskew_max_deg))
    if cfg.ink_crop_horizontal:
        gray = horizontal_ink_crop(gray)
    if cfg.clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    h, w = gray.shape[:2]
    tw, th = target_size(w, h, cfg, scale)
    interp = cv2.INTER_AREA if (tw < w) else cv2.INTER_CUBIC
    return cv2.resize(gray, (tw, th), interpolation=interp)


def load_line(image_id: str, cfg: Config = None, scale: float = 1.0,
              use_cache: bool = True) -> np.ndarray:
    """Normalised grayscale line image, cached at scale 1.0."""
    cfg = cfg or CFG
    if use_cache and abs(scale - 1.0) < 1e-6:
        cached = CACHE_DIR / f"{image_id}.png"
        if cached.exists():
            img = cv2.imread(str(cached), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
    path = image_path(image_id)
    if path is None:
        return np.full((cfg.line_height, cfg.min_line_width), 235, np.uint8)
    out = normalize_line(read_gray(path), cfg, scale)
    if use_cache and abs(scale - 1.0) < 1e-6:
        cv2.imwrite(str(CACHE_DIR / f"{image_id}.png"), out)
    return out


def to_pil(gray: np.ndarray):
    return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

# ### 6.1 Warm the cache
#
# Normalisation costs ~15 ms per image and is deterministic, so it is done once for
# all 5 472 crops. Training then reads small PNGs, which keeps the dataloader ahead
# of an 8B model even at batch 16.

from concurrent.futures import ThreadPoolExecutor

def warm_cache(ids: Sequence[str], workers: int = 12) -> None:
    todo = [i for i in ids if not (CACHE_DIR / f"{i}.png").exists()]
    if not todo:
        print(f"[cache] {CACHE_DIR.name}: complete")
        return
    print(f"[cache] normalising {len(todo)} images -> {CACHE_DIR}")
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        list(progress(ex.map(lambda i: load_line(i), todo), total=len(todo),
                      desc="cache", unit="img"))
    print(f"[cache] done in {time.time()-t0:.0f}s")


stage("image normalisation cache")
warm_cache(list(train_df.ID) + list(test_df.ID))

_sizes = []
for i in train_df.ID.sample(400, random_state=0):
    im = load_line(i)
    _sizes.append((im.shape[1], im.shape[0], (im.shape[1] // CFG.patch_grid) * (im.shape[0] // CFG.patch_grid)))
_sizes = np.array(_sizes)
print(f"[cache] normalised width  median {np.median(_sizes[:,0]):.0f} p90 {np.percentile(_sizes[:,0],90):.0f}")
print(f"[cache] normalised height median {np.median(_sizes[:,1]):.0f}")
print(f"[cache] visual tokens     median {np.median(_sizes[:,2]):.0f} p90 {np.percentile(_sizes[:,2],90):.0f} max {_sizes[:,2].max()}")

fig, axes = plt.subplots(4, 1, figsize=(16, 6))
for ax_, (_, row) in zip(axes, train_df.sample(4, random_state=11).iterrows()):
    ax_.imshow(load_line(row.ID), cmap="gray")
    ax_.set_title(row.Target, fontsize=8, loc="left")
    ax_.axis("off")
plt.tight_layout()
plt.savefig(OUT / "eda_normalised.png", dpi=110)
plt.close(fig)
print(f"[cache] preview -> {OUT/'eda_normalised.png'}")

# ### 6.2 Augmentation
#
# Handwriting augmentation has to change *how* a glyph is written without changing
# *which* glyph it is. The stack below is deliberately mild on geometry (±1.5°
# rotation, ±0.06 shear) and more aggressive on the degradation axes that actually
# vary across the archive: ink weight, paper contrast, blur, noise and stains.

def _rand_affine(img: np.ndarray, rng: np.random.Generator, cfg: Config) -> np.ndarray:
    h, w = img.shape[:2]
    angle = rng.uniform(-cfg.aug_rotate_deg, cfg.aug_rotate_deg)
    shear = rng.uniform(-cfg.aug_shear, cfg.aug_shear)
    sx = 1.0 + rng.uniform(-cfg.aug_scale_jitter, cfg.aug_scale_jitter)
    sy = 1.0 + rng.uniform(-cfg.aug_scale_jitter, cfg.aug_scale_jitter) * 0.5
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 0] *= sx
    M[0, 1] += shear
    M[1, 1] *= sy
    M[0, 2] += rng.uniform(-2, 2)
    M[1, 2] += rng.uniform(-2, 2)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _elastic(img: np.ndarray, rng: np.random.Generator, alpha=1.6, sigma=0.12) -> np.ndarray:
    """Low-frequency wobble along the baseline - mimics natural hand tremor."""
    h, w = img.shape[:2]
    gh, gw = 3, max(4, w // max(h, 1) + 2)
    dy = rng.normal(0, alpha, (gh, gw)).astype(np.float32) * h * sigma * 0.25
    dx = rng.normal(0, alpha, (gh, gw)).astype(np.float32) * h * sigma * 0.15
    dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_CUBIC)
    dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_CUBIC)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float32)
    f = f * rng.uniform(0.85, 1.15) + rng.uniform(-18, 18)          # brightness/contrast
    f = np.clip(f, 0, 255)
    gamma = rng.uniform(0.75, 1.35)
    f = 255.0 * np.power(f / 255.0, gamma)
    if rng.random() < 0.35:                                          # illumination ramp
        w = img.shape[1]
        ramp = np.linspace(rng.uniform(-15, 15), rng.uniform(-15, 15), w, dtype=np.float32)
        f = f + ramp[None, :]
    return np.clip(f, 0, 255).astype(np.uint8)


def _morph(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = np.ones((2, 2), np.uint8) if rng.random() < 0.7 else np.ones((3, 3), np.uint8)
    # ink is dark: erode() thickens strokes, dilate() thins them
    return cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)


def _noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float32)
    if rng.random() < 0.6:
        f += rng.normal(0, rng.uniform(2, 9), img.shape).astype(np.float32)
    if rng.random() < 0.5:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.4, 1.3))
    if rng.random() < 0.25:                                          # JPEG ringing
        q = int(rng.integers(35, 85))
        ok, enc = cv2.imencode(".jpg", np.clip(f, 0, 255).astype(np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            f = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


def _stains(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(int(rng.integers(1, 4))):
        r = int(rng.integers(max(3, h // 12), max(5, h // 3)))
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        layer = np.zeros_like(out, dtype=np.float32)
        cv2.circle(layer, (cx, cy), r, float(rng.uniform(18, 55)), -1)
        layer = cv2.GaussianBlur(layer, (0, 0), r * 0.6)
        out = np.clip(out.astype(np.float32) - layer, 0, 255).astype(np.uint8)
    return out


def augment_line(img: np.ndarray, rng: np.random.Generator, cfg: Config = None) -> np.ndarray:
    cfg = cfg or CFG
    if rng.random() > cfg.aug_prob:
        return img
    out = img
    out = _rand_affine(out, rng, cfg)
    if rng.random() < cfg.aug_elastic_prob:
        out = _elastic(out, rng)
    if rng.random() < cfg.aug_photometric_prob:
        out = _photometric(out, rng)
    if rng.random() < cfg.aug_morph_prob:
        out = _morph(out, rng)
    if rng.random() < cfg.aug_noise_prob:
        out = _noise(out, rng)
    if rng.random() < cfg.aug_cutout_prob:
        out = _stains(out, rng)
    return out


_rng = np.random.default_rng(0)
_demo_id = train_df.ID.iloc[5]
_base = load_line(_demo_id)
fig, axes = plt.subplots(6, 1, figsize=(16, 8))
axes[0].imshow(_base, cmap="gray"); axes[0].set_title("normalised (no augmentation)", fontsize=8, loc="left")
for ax_ in axes[1:]:
    ax_.imshow(augment_line(_base, _rng), cmap="gray")
for ax_ in axes:
    ax_.axis("off")
plt.tight_layout()
plt.savefig(OUT / "aug_preview.png", dpi=110)
plt.close(fig)
print(f"[aug] preview -> {OUT/'aug_preview.png'}")

# ==========================================================================
# p07_stackmix.py
# ==========================================================================

# ## 7. StackMix — word-crop recombination
#
# **Permitted by the organisers** (discussion #34514): segmenting crops out of the
# provided training images and programmatically recombining them into new labelled
# lines counts as training-data augmentation, as long as the process is fully
# automated and reproducible and no test images, external images or manual
# annotation are involved. Everything below reads only `Train.csv` images.
#
# ### Why the quality gates are strict
#
# The text is cursive and words are frequently ligatured, so a naive projection-profile
# split is wrong more often than it is right. Measured on 600 random training lines,
# an unfiltered splitter matches the true word count on ~27% of lines and
# under-segments badly on the rest. Two gates fix this:
#
# * **gap margin** — merge column runs until exactly `n_words` segments remain, then
#   require the smallest surviving separator to be ≥ `stackmix_margin`× the largest
#   gap that had to be merged. Ambiguous splits are rejected.
# * **width/length correlation** — the widths of the segments must correlate with the
#   character lengths of the words at ρ ≥ `stackmix_corr`. This catches the failure
#   mode where the band detector locks onto the wrong text line entirely.
#
# Together they accept ~7% of lines. That is a small bank, but it is a *clean* one,
# and a mislabelled synthetic line is far more damaging than a missing one.
#
# ### Closing the domain gap
#
# Real crops always carry slivers of the neighbouring lines and real parchment
# texture. A synthetic line pasted onto white paper would be trivially separable from
# the real distribution, so instead each synthetic line is composited into a **real
# background donor**: a training crop whose centre band has been inpainted away,
# keeping its paper, its stains, and its neighbouring-line fragments. Words are
# transferred by subtracting their ink darkness, which leaves no seams.

@dataclass
class WordCrop:
    img: np.ndarray      # grayscale patch covering the word plus vertical padding
    band_h: int          # height of the main-line band it came from
    core_h: float        # x-height of the source line (the real size reference)
    pad_top: int         # rows of padding above the band inside `img`
    pad_x: int           # columns of blank padding on each side of the ink
    stroke: float        # mean stroke width / x-height, a proxy for the hand


@dataclass
class Donor:
    bg: np.ndarray       # crop with the centre band inpainted away
    band: tuple          # (top, bottom) of the erased band
    core_h: float        # x-height of the line that was erased
    stroke: float        # stroke style of the scribe who wrote it
    gaps: np.ndarray     # inter-word gaps, as a fraction of the x-height


def line_style(binary: np.ndarray, band: tuple) -> tuple[float, float]:
    """(x-height, stroke width / x-height) of the main line.

    Band height is dominated by ascenders and descenders and is a poor size
    reference; the x-height - the rows carrying most of the ink - is what makes two
    hands look the same size. Stroke width comes from the mean length of horizontal
    ink runs, which separates a fine quill from a broad one.
    """
    top, bot = band
    strip = binary[top:bot] > 0
    if strip.size == 0 or not strip.any():
        return float(max(bot - top, 1)), 1.0
    rows = strip.sum(axis=1).astype(np.float32)
    core_h = float(max((rows > 0.5 * rows.max()).sum(), 1))
    ink = int(strip.sum())
    starts = int((strip[:, 1:] & ~strip[:, :-1]).sum() + strip[:, 0].sum())
    stroke = ink / max(starts, 1)
    return core_h, stroke / core_h


def main_line_band(binary: np.ndarray, min_frac: float = 0.35) -> tuple[int, int]:
    """Vertical extent of the centre line, ignoring neighbour slivers."""
    h = binary.shape[0]
    prof = binary.sum(axis=1).astype(np.float32) / 255.0
    k = max(3, int(round(h * 0.06)) | 1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, k), 0).ravel()
    if prof.max() <= 0:
        return 0, h
    weights = np.exp(-0.5 * ((np.arange(h) - h / 2) / (0.28 * h)) ** 2)
    seed = int(np.argmax(prof * weights))
    thr = prof.max() * min_frac
    top = seed
    while top > 0 and prof[top - 1] > thr:
        top -= 1
    bot = seed
    while bot < h - 1 and prof[bot + 1] > thr:
        bot += 1
    return top, bot + 1


def column_runs(binary: np.ndarray, band: tuple) -> list:
    top, bot = band
    band_h = max(bot - top, 1)
    col = binary[top:bot].sum(axis=0).astype(np.float32) / 255.0
    k = max(3, int(round(band_h * 0.05)) | 1)
    col = cv2.GaussianBlur(col.reshape(1, -1), (k, 1), 0).ravel()
    on = col > max(0.04 * band_h, 0.6)
    runs, i, w = [], 0, len(on)
    while i < w:
        if on[i]:
            j = i
            while j < w and on[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    return runs


def merge_to_n(runs: list, n: int) -> tuple[list | None, float, list]:
    """Merge across the smallest gaps until `n` segments remain."""
    if len(runs) < n or n < 1:
        return None, 0.0, []
    segs = [list(r) for r in runs]
    merged_max = 0.0
    while len(segs) > n:
        gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
        idx = int(np.argmin(gaps))
        merged_max = max(merged_max, gaps[idx])
        segs[idx][1] = segs[idx + 1][1]
        del segs[idx + 1]
    seps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
    margin = (min(seps) + 1.0) / (merged_max + 1.0) if seps else 1e9
    return [tuple(s) for s in segs], float(margin), seps


def segment_line(img: np.ndarray, words: Sequence[str], cfg: Config):
    """Return (segments, band, gaps) when the split passes both quality gates."""
    if len(words) < 3:
        return None
    binary = ink_mask(img)
    band = main_line_band(binary)
    if band[1] - band[0] < 8:
        return None
    segs, margin, seps = merge_to_n(column_runs(binary, band), len(words))
    if segs is None or margin < cfg.stackmix_margin:
        return None
    widths = np.array([b - a for a, b in segs], dtype=np.float64)
    lengths = np.array([len(w) for w in words], dtype=np.float64)
    if widths.std() < 1e-6 or lengths.std() < 1e-6:
        return None
    corr = float(np.corrcoef(widths, lengths)[0, 1])
    if not np.isfinite(corr) or corr < cfg.stackmix_corr:
        return None
    return segs, band, np.array(seps, dtype=np.float32)


def build_word_bank(df: pd.DataFrame, cfg: Config, max_lines: int | None = None):
    """Segment training lines into a bank of labelled word crops + background donors."""
    bank: dict[str, list[WordCrop]] = defaultdict(list)
    donors: list[Donor] = []
    rows = df if max_lines is None else df.head(max_lines)
    accepted = 0
    bar = progress(rows.itertuples(), total=len(rows), desc="stackmix scan", unit="line")
    for row in bar:
        path = image_path(row.ID)
        if path is None:
            continue
        # Segment at native resolution: the normalised copies are upscaled, which
        # blurs the inter-word gaps the splitter relies on.
        img = flatten_background(read_gray(path))
        if img.shape[0] < 36:
            continue
        words = row.Target.split()
        seg = segment_line(img, words, cfg)
        if seg is None:
            continue
        segs, band, gaps = seg
        accepted += 1
        bar.set_postfix_str(f"accepted {accepted}", refresh=False)
        top, bot = band
        band_h = bot - top
        binary = ink_mask(img)
        core_h, stroke = line_style(binary, band)
        pad = int(band_h * 0.3)
        y0, y1 = max(top - pad, 0), min(bot + pad, img.shape[0])
        m = max(1, int(band_h * 0.03))
        for (x0, x1), word in zip(segs, words):
            xa, xb = max(x0 - m, 0), min(x1 + m, img.shape[1])
            bank[word].append(WordCrop(img[y0:y1, xa:xb].copy(), band_h, core_h,
                                       top - y0, m, stroke))

        # Background donor: erase the centre band's ink, keep everything else.
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[top:bot] = binary[top:bot]
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        bg = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        donors.append(Donor(bg, (top, bot), core_h, stroke, gaps / max(core_h, 1)))

    total_crops = sum(len(v) for v in bank.values())
    print(f"[stackmix] accepted {accepted}/{len(rows)} lines ({accepted/max(len(rows),1):.1%})")
    print(f"[stackmix] bank: {total_crops} crops over {len(bank)} word types, {len(donors)} donors")
    return dict(bank), donors


class StackMixRenderer:
    """Composites new labelled lines out of banked word crops and real backgrounds."""

    def __init__(self, bank: dict, donors: list, corpus: Sequence[str], cfg: Config,
                 aspect_per_char: np.ndarray | None = None):
        self.bank = bank
        self.donors = donors
        self.cfg = cfg
        # w / (h * n_chars) over the real crops. Donor lines are a biased sample -
        # the splitter only accepts well-spaced hands - so their gaps alone would
        # render systematically airier lines than the test set contains.
        self.apc = (np.asarray(aspect_per_char, dtype=np.float64)
                    if aspect_per_char is not None and len(aspect_per_char) else None)
        if self.apc is not None:
            self.apc = self.apc[np.isfinite(self.apc) & (self.apc > 0)]
        self.words = sorted(bank)
        self.word_set = set(self.words)
        freq = np.array([max(WORD_FREQ.get(w, 1), 1) for w in self.words], dtype=np.float64)
        self.unigram_p = freq / freq.sum()
        self.bigram: dict[str, list[str]] = defaultdict(list)
        for line in corpus:
            toks = line.split()
            for a, b in zip(toks, toks[1:]):
                if a in self.word_set and b in self.word_set:
                    self.bigram[a].append(b)
        self.corpus_lines = [ln.split() for ln in corpus]
        gaps = np.concatenate([d.gaps for d in donors if d.gaps.size]) if donors else np.array([0.4])
        self.gap_lo, self.gap_hi = np.percentile(gaps, [25, 75])
        self.gap_lo = max(float(self.gap_lo), 0.1)
        self.gap_hi = max(float(self.gap_hi), self.gap_lo + 0.1)
        self.nw_values, self.nw_p = self._length_prior()

    def _length_prior(self) -> tuple[np.ndarray, np.ndarray]:
        """Word counts drawn from the real lines, not from a flat range.

        The metric weights long references more heavily, so the useful synthetic
        lines are the long ones; sampling the empirical head of the distribution
        above `stackmix_min_words` puts the augmentation where the score is.
        """
        lo, hi = self.cfg.stackmix_min_words, self.cfg.stackmix_max_words
        counts = np.array([len(t) for t in self.corpus_lines], dtype=np.int64)
        counts = counts[(counts >= lo) & (counts <= hi)]
        if counts.size < 20:
            values = np.arange(lo, hi + 1)
            return values, np.full(values.size, 1.0 / values.size)
        values, freq = np.unique(counts, return_counts=True)
        return values, freq / freq.sum()

    # ----------------------------------------------------------------- text --
    def sample_words(self, rng: np.random.Generator) -> list[str]:
        n = int(rng.choice(self.nw_values, p=self.nw_p))
        if rng.random() < self.cfg.stackmix_ngram_prob:
            span = self._corpus_span(rng, n)
            if span:
                return span
        return self._markov(rng, n)

    def _corpus_span(self, rng: np.random.Generator, n: int) -> list[str] | None:
        for _ in range(12):
            toks = self.corpus_lines[int(rng.integers(len(self.corpus_lines)))]
            if len(toks) < 3:
                continue
            k = min(n, len(toks))
            s = int(rng.integers(0, len(toks) - k + 1))
            span = toks[s:s + k]
            if all(w in self.word_set for w in span):
                if len(span) < n:  # splice a second span from another line
                    tail = self._corpus_span(rng, n - len(span))
                    if tail:
                        span = span + tail
                return span
        return None

    def _markov(self, rng: np.random.Generator, n: int) -> list[str]:
        out = [self.words[int(rng.choice(len(self.words), p=self.unigram_p))]]
        while len(out) < n:
            succ = self.bigram.get(out[-1])
            if succ and rng.random() < 0.75:
                out.append(succ[int(rng.integers(len(succ)))])
            else:
                out.append(self.words[int(rng.choice(len(self.words), p=self.unigram_p))])
        return out

    # ---------------------------------------------------------------- image --
    def _background(self, donor: Donor, width: int, rng: np.random.Generator) -> np.ndarray:
        """Mirror-tile one donor horizontally so its neighbour lines stay coherent."""
        bg = donor.bg
        if bg.shape[1] >= width:
            off = int(rng.integers(0, bg.shape[1] - width + 1)) if bg.shape[1] > width else 0
            return bg[:, off:off + width].copy()
        reps = int(np.ceil(width / bg.shape[1])) + 1
        tiles = [bg if k % 2 == 0 else bg[:, ::-1] for k in range(reps)]
        return np.concatenate(tiles, axis=1)[:, :width].copy()

    def _pick_crop(self, word: str, style: float, rng: np.random.Generator) -> WordCrop | None:
        """Prefer a crop written in a hand similar to the donor's."""
        pool = self.bank.get(word)
        if not pool:
            return None
        if len(pool) <= 2:
            return pool[int(rng.integers(len(pool)))]
        order = np.argsort([abs(c.stroke - style) for c in pool])
        k = max(1, min(3, len(pool)))
        return pool[int(order[int(rng.integers(k))])]

    def render(self, rng: np.random.Generator) -> tuple[np.ndarray, str] | None:
        words = self.sample_words(rng)
        donor = self.donors[int(rng.integers(len(self.donors)))]
        crops = []
        for w in words:
            c = self._pick_crop(w, donor.stroke, rng)
            if c is None:
                return None
            crops.append(c)

        height = donor.bg.shape[0]
        band_top, band_bot = donor.band
        target_core = max(donor.core_h, 4.0)
        # Let the word count set the width, as it does on a real line, and cap only
        # at the widest aspect the corpus actually contains. Capping tighter than
        # this silently truncates the line and shortens every synthetic label.
        max_width = int(height * 34.0)

        # Scale every crop to the donor's x-height, then plan the layout on paper
        # before touching a pixel, so the whole line can be resized in one pass.
        iso = [(target_core / max(c.core_h, 1.0)) * float(rng.uniform(0.96, 1.04)) for c in crops]
        widths = [max(int(round(c.img.shape[1] * s)), 4) for c, s in zip(crops, iso)]
        pads = [c.pad_x * s for c, s in zip(crops, iso)]
        # Gaps are measured ink-to-ink, but each crop already carries blank padding,
        # so the padding of both neighbours has to come out of the gap.
        raw_gaps = [max(1.0, rng.uniform(self.gap_lo, self.gap_hi) * target_core
                        - pads[k] - pads[k + 1]) for k in range(len(crops) - 1)]
        margin = max(2, int(target_core * rng.uniform(0.1, 0.8)))

        # Give the line the width a real line of this many characters would have.
        # The donor pool is biased towards broad, generously spaced hands - those
        # are the ones the splitter can segment - so without this the synthetic set
        # sits to the right of the real aspect distribution and teaches the model a
        # geometry the test set does not contain. The correction is split between a
        # mild horizontal squeeze of the glyphs and the inter-word spacing; both are
        # clipped to ranges the real data already spans.
        xs, gap_scale = 1.0, 1.0
        if self.apc is not None:
            n_chars = len(" ".join(words))
            need = height * float(self.apc[int(rng.integers(self.apc.size))]) * n_chars - 2 * margin
            have = sum(widths) + sum(raw_gaps)
            xs = float(np.clip(need / max(have, 1.0), 0.82, 1.14))
            if raw_gaps:
                gap_scale = float(np.clip((need - sum(widths) * xs) / sum(raw_gaps), 0.4, 1.6))

        scaled = []
        for c, s in zip(crops, iso):
            nh = max(int(round(c.img.shape[0] * s)), 4)
            nw = max(int(round(c.img.shape[1] * s * xs)), 4)
            scaled.append((cv2.resize(c.img, (nw, nh), interpolation=cv2.INTER_LINEAR),
                           int(round(c.pad_top * s))))
        gaps = [max(1, int(round(g * gap_scale))) for g in raw_gaps]
        width = min(margin * 2 + sum(s[0].shape[1] for s in scaled) + sum(gaps), max_width)
        canvas = self._background(donor, int(width), rng)

        # One ink level for the whole line: a line is written with one pen, and
        # per-word darkness jitter is the giveaway that a crop is synthetic.
        ink_gain = float(rng.uniform(0.82, 1.0))
        x = margin
        used = 0
        for k, (im, pad_top) in enumerate(scaled):
            if x + im.shape[1] > canvas.shape[1] - 2:
                break
            if im.shape[0] >= height:
                im = im[:max(height - 1, 1)]
            y = int(np.clip(band_top - pad_top, 0, height - im.shape[0] - 1))
            crop_paper = float(np.percentile(im, 85))
            ink = np.clip(crop_paper - im.astype(np.float32), 0, 255)
            peak = float(ink.max())
            if peak > 20:                      # equalise contrast across scribes
                ink *= min(1.6, 150.0 / peak)
            region = canvas[y:y + im.shape[0], x:x + im.shape[1]].astype(np.float32)
            blended = np.clip(region - ink * ink_gain, 0, 255)
            canvas[y:y + im.shape[0], x:x + im.shape[1]] = blended.astype(np.uint8)
            x += im.shape[1] + (gaps[k] if k < len(gaps) else 0)
            used = k + 1

        if used < max(3, self.cfg.stackmix_min_words // 2):
            return None
        canvas = canvas[:, :int(np.clip(x + margin, 8, canvas.shape[1]))]
        if self.cfg.clahe:
            canvas = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(canvas)
        tw, th = target_size(canvas.shape[1], canvas.shape[0], self.cfg)
        interp = cv2.INTER_AREA if tw < canvas.shape[1] else cv2.INTER_CUBIC
        canvas = cv2.resize(canvas, (tw, th), interpolation=interp)
        return canvas, " ".join(words[:used])

STACKMIX = None
if CFG.stackmix:
    stage("StackMix word bank")
    set_seed(CFG.stackmix_seed)
    _bank, _donors = build_word_bank(train_df, CFG)
    if sum(len(v) for v in _bank.values()) >= 300 and len(_donors) >= 20:
        _apc = (tr_stats["w"] / (tr_stats["h"] * tr_stats["n_chars"].clip(lower=1))).values
        STACKMIX = StackMixRenderer(_bank, _donors, train_df["Target"].tolist(), CFG, _apc)
        rng = np.random.default_rng(CFG.stackmix_seed)
        fig, axes = plt.subplots(6, 1, figsize=(16, 8))
        shown = 0
        for ax_ in axes:
            out = None
            for _ in range(20):
                out = STACKMIX.render(rng)
                if out is not None:
                    break
            if out is None:
                ax_.axis("off")
                continue
            im, txt = out
            ax_.imshow(im, cmap="gray")
            ax_.set_title(txt, fontsize=8, loc="left")
            ax_.axis("off")
            shown += 1
        plt.tight_layout()
        plt.savefig(OUT / "stackmix_preview.png", dpi=110)
        plt.close(fig)
        print(f"[stackmix] rendered {shown} previews -> {OUT/'stackmix_preview.png'}")

        # Distribution check: synthetic geometry and label length must sit inside
        # the real ranges, otherwise the model learns a shortcut instead of a script.
        stats = []
        for _ in range(150):
            out = STACKMIX.render(rng)
            if out is not None:
                im, txt = out
                stats.append((im.shape[1] / max(im.shape[0], 1), len(txt), len(txt.split())))
        if stats:
            syn = np.array(stats)
            real_aspect = (tr_stats.w / tr_stats.h).values
            print(f"[stackmix] aspect  synthetic med {np.median(syn[:,0]):.1f} "
                  f"[p10 {np.percentile(syn[:,0],10):.1f}, p90 {np.percentile(syn[:,0],90):.1f}]"
                  f"  | real med {np.median(real_aspect):.1f} "
                  f"[p10 {np.percentile(real_aspect,10):.1f}, p90 {np.percentile(real_aspect,90):.1f}]")
            print(f"[stackmix] chars   synthetic med {np.median(syn[:,1]):.0f} "
                  f"| real med {train_df.Target.str.len().median():.0f}")
            print(f"[stackmix] words   synthetic med {np.median(syn[:,2]):.0f} "
                  f"| real med {train_df.Target.str.split().apply(len).median():.0f}")
        print("[stackmix] INSPECT THE PREVIEW before a long run. If the synthetic lines do "
              "not look like the real ones, set CFG.stackmix = False.")
    else:
        print("[stackmix] bank too small after quality gating - disabling")
        CFG.stackmix = False
else:
    print("[stackmix] disabled")

# ==========================================================================
# p08_dataset.py
# ==========================================================================

# ## 8. Prompt, dataset and collator
#
# ### Prompt
#
# The prompt encodes three facts the model cannot infer from a single example: the
# label covers **only the centre line** (crops carry clipped neighbours), the original
# early-modern spelling must be preserved rather than modernised, and `^` is a
# meaningful superscript marker (`y^e`, `15^th`, `w^tsoever`). The identical string is
# used at training and at inference — a mismatch there is one of the most common
# silent regressions in VLM fine-tuning.
#
# ### Loss masking
#
# Only the assistant's transcription contributes to the loss. Rather than
# string-matching the answer inside the token stream (which breaks whenever the
# tokenizer merges a boundary character), the assistant header is derived from the
# chat template itself and located as a token subsequence, so masking is exact for any
# Qwen chat template.

OCR_PROMPT = (
    "This is a cropped line from a 17th-19th century handwritten Barbadian legal record "
    "(deed, will, inventory or census page).\n"
    "Transcribe the single complete line of handwriting that runs across the middle of the image. "
    "Partial lines clipped at the top and bottom edges are context only - do not transcribe them.\n"
    "Reproduce the original early-modern spelling, capitalisation and punctuation exactly as "
    "written; do not modernise or correct it. Keep the caret '^' that marks a superscript "
    "(for example y^e, 15^th, w^tsoever).\n"
    "Reply with the transcription of that line and nothing else."
)


def build_messages(with_answer: str | None = None) -> list:
    msgs = [{
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": OCR_PROMPT}],
    }]
    if with_answer is not None:
        msgs.append({"role": "assistant", "content": [{"type": "text", "text": with_answer}]})
    return msgs


def find_last_subsequence(hay: list, needle: list) -> int:
    """Index one past the last occurrence of `needle` in `hay`, or -1."""
    n, m = len(hay), len(needle)
    if m == 0 or m > n:
        return -1
    for start in range(n - m, -1, -1):
        if hay[start:start + m] == needle:
            return start + m
    return -1


class LineDataset(torch.utils.data.Dataset):
    """Real training lines plus, optionally, freshly rendered StackMix lines."""

    def __init__(self, df: pd.DataFrame, cfg: Config, train: bool,
                 renderer: "StackMixRenderer | None" = None, seed: int = 0):
        self.ids = df["ID"].tolist()
        self.texts = df["Target"].tolist()
        self.cfg = cfg
        self.train = train
        self.renderer = renderer if train else None
        self.seed = seed
        self.epoch = 0
        self.n_real = len(self.ids)
        self.n_synth = int(self.n_real * cfg.stackmix_ratio) if self.renderer else 0
        real_len = [self._estimate_tokens(i, t) for i, t in zip(self.ids, self.texts)]
        med = int(np.median(real_len)) if real_len else 400
        self.lengths = real_len + [med] * self.n_synth

    def _estimate_tokens(self, image_id: str, text: str) -> int:
        p = CACHE_DIR / f"{image_id}.png"
        if p.exists():
            try:
                with Image.open(p) as im:
                    w, h = im.size
                return (w // self.cfg.patch_grid) * (h // self.cfg.patch_grid) + len(text) // 3
            except Exception:
                pass
        return 400 + len(text) // 3

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n_real + self.n_synth

    def __getitem__(self, idx: int) -> dict:
        if idx < self.n_real:
            img = load_line(self.ids[idx], self.cfg)
            text = self.texts[idx]
            if self.train:
                rng = np.random.default_rng((self.seed, self.epoch, idx))
                img = augment_line(img, rng, self.cfg)
        else:
            rng = np.random.default_rng((self.seed, self.epoch, idx, 7919))
            out = None
            for _ in range(8):
                out = self.renderer.render(rng)
                if out is not None:
                    break
            if out is None:  # fall back to a real line rather than skipping a slot
                j = int(rng.integers(self.n_real))
                img, text = load_line(self.ids[j], self.cfg), self.texts[j]
            else:
                img, text = out
            img = augment_line(img, rng, self.cfg)
        return {"image": to_pil(img), "text": text}


class OCRCollator:
    """Chat-template batching with exact assistant-only loss masking."""

    def __init__(self, processor, train: bool = True, max_length: int = 4096):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.train = train
        self.max_length = max_length
        probe_nogen = processor.apply_chat_template(build_messages(), tokenize=False,
                                                    add_generation_prompt=False)
        probe_gen = processor.apply_chat_template(build_messages(), tokenize=False,
                                                  add_generation_prompt=True)
        suffix = probe_gen[len(probe_nogen):] if probe_gen.startswith(probe_nogen) else "assistant\n"
        self.assistant_header = suffix
        self.header_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        if not self.header_ids:
            raise RuntimeError("Could not derive the assistant header from the chat template")

    def __call__(self, examples: list) -> dict:
        images = [ex["image"] for ex in examples]
        texts = [
            self.processor.apply_chat_template(
                build_messages(ex["text"] if self.train else None),
                tokenize=False, add_generation_prompt=not self.train,
            )
            for ex in examples
        ]
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        if not self.train:
            return batch

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        pad_id = self.tokenizer.pad_token_id
        for i in range(input_ids.size(0)):
            seq = input_ids[i].tolist()
            cut = find_last_subsequence(seq, self.header_ids)
            if cut < 0:
                raise RuntimeError("Assistant header not found - chat template mismatch")
            labels[i, :cut] = -100
        if pad_id is not None:
            labels[input_ids == pad_id] = -100
        for name in ("image_token_id", "video_token_id", "vision_start_token_id"):
            tok = getattr(self.processor, name, None) or getattr(
                getattr(self.processor, "tokenizer", None), name, None)
            if isinstance(tok, int):
                labels[input_ids == tok] = -100
        batch["labels"] = labels
        return batch

# ==========================================================================
# p09_model.py
# ==========================================================================

# ## 9. Model
#
# `AutoModelForImageTextToText` is used rather than a family-specific class so the
# same code runs on Qwen3-VL, Qwen2.5-VL and Qwen2-VL. Attention falls back
# gracefully: FlashAttention-2 if the wheel imports, otherwise PyTorch SDPA.
#
# LoRA targets are **discovered from the graph** instead of hard-coded suffix lists.
# That matters because the projector and vision blocks are named differently in every
# release, and for handwriting the vision tower is exactly where the adaptation needs
# to happen — the language prior already knows 17th-century legalese, the encoder is
# what has never seen secretary hand.
#
# `build_model` serves both fine-tuning modes and both directions of the checkpoint
# round-trip. `CFG.finetune_mode = "full"` skips PEFT entirely and unfreezes the
# network, honouring `full_freeze_vision` / `full_freeze_llm`; on reload the
# checkpoint directory is inspected rather than assumed, so a full checkpoint
# replaces the backbone while an adapter is layered on top of it. That is what lets
# the same decoding and GRPO code run unchanged whichever mode produced the weights.

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training


def resolve_attn(cfg: Config) -> str:
    if cfg.attn_implementation != "auto":
        return cfg.attn_implementation
    try:
        import flash_attn  # noqa: F401

        major = torch.cuda.get_device_properties(0).major if torch.cuda.is_available() else 0
        if major >= 8:
            return "flash_attention_2"
    except Exception:
        pass
    return "sdpa"


def load_processor(cfg: Config, padding_side: str = "right"):
    processor = AutoProcessor.from_pretrained(cfg.model_id, trust_remote_code=True)
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        # Our own resizing already snapped to the patch grid; widen the processor's
        # limits so its smart_resize becomes a no-op and the token count is exactly
        # what Section 6 predicted. Note that for Qwen VL processors the entries of
        # `size` are pixel *counts*, not edge lengths.
        max_px = cfg.line_height * 2 * cfg.max_line_width
        min_px = cfg.min_line_height * cfg.min_line_width
        for attr, val in (("min_pixels", min_px), ("max_pixels", max_px)):
            if hasattr(ip, attr):
                setattr(ip, attr, val)
        if isinstance(getattr(ip, "size", None), dict):
            ip.size = {**ip.size, "shortest_edge": min_px, "longest_edge": max_px}
        grid = int(getattr(ip, "patch_size", 16)) * int(getattr(ip, "merge_size", 2))
        if grid != cfg.patch_grid:
            print(f"!! processor tokenises on a {grid}px grid but the image cache was "
                  f"built for {cfg.patch_grid}px. Set CFG.patch_grid = {grid}, delete "
                  f"{CACHE_DIR}, and re-run Section 6.")
    processor.tokenizer.padding_side = padding_side
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    return processor


def _from_pretrained(cfg: Config, quant, attn, device_map, source: str | None = None):
    source = source or cfg.model_id
    kwargs = dict(attn_implementation=attn, quantization_config=quant,
                  device_map=device_map, trust_remote_code=True)
    try:                       # transformers >= 4.56
        return AutoModelForImageTextToText.from_pretrained(source, dtype=torch.bfloat16, **kwargs)
    except TypeError:          # older signature
        return AutoModelForImageTextToText.from_pretrained(source, torch_dtype=torch.bfloat16, **kwargs)


def is_adapter_dir(path: str | Path) -> bool:
    """A PEFT adapter ships a config; a full checkpoint ships model weights."""
    return (Path(path) / "adapter_config.json").exists()


def set_trainable_towers(model, cfg: Config) -> tuple[int, int]:
    """Full fine-tune: unfreeze everything, then re-freeze the towers we opted out of."""
    frozen = 0
    for name, param in model.named_parameters():
        low = name.lower()
        is_merger = "merger" in low or "projector" in low
        is_vision = ("visual" in low or "vision" in low) and not is_merger
        keep = not ((is_vision and cfg.full_freeze_vision)
                    or (not is_vision and not is_merger and cfg.full_freeze_llm))
        param.requires_grad_(keep)
        if not keep:
            frozen += param.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, frozen


def find_lora_targets(model, cfg: Config) -> list:
    """Every adaptable Linear, filtered by which tower we want to train."""
    import torch.nn as nn

    try:
        import bitsandbytes as bnb

        linear_types = (nn.Linear, bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)
    except Exception:
        linear_types = (nn.Linear,)

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, linear_types):
            continue
        low = name.lower()
        if "lm_head" in low or "embed" in low or "head.out" in low:
            continue
        is_merger = "merger" in low or "projector" in low or "mm_projector" in low
        is_vision = ("visual" in low or "vision" in low) and not is_merger
        if is_vision and not cfg.train_vision_tower:
            continue
        if is_merger and not cfg.train_merger:
            continue
        targets.append(name)
    return targets


def build_model(cfg: Config, adapter_path: str | Path | None = None,
                for_training: bool = True, verbose: bool = True):
    quant = None
    if cfg.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    attn = resolve_attn(cfg)
    device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else "cpu"

    # A full fine-tune saves whole weights, so a checkpoint replaces the backbone
    # rather than being layered on top of it.
    full_ckpt = (adapter_path is not None and not is_adapter_dir(adapter_path))
    t0 = time.time()
    model = _from_pretrained(cfg, quant, attn, device_map,
                             source=str(adapter_path) if full_ckpt else None)
    if verbose:
        src = f"{adapter_path} (full weights)" if full_ckpt else cfg.model_id
        print(f"[model] {src} | attn={attn} | 4bit={cfg.load_in_4bit} "
              f"| loaded in {time.time()-t0:.0f}s")

    if adapter_path is not None and not full_ckpt:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=for_training)
        if verbose:
            print(f"[model] adapter <- {adapter_path}")
    elif for_training and cfg.finetune_mode == "full":
        trainable, frozen = set_trainable_towers(model, cfg)
        if verbose:
            note = [n for n, on in (("vision", cfg.full_freeze_vision),
                                    ("llm", cfg.full_freeze_llm)) if on]
            print(f"[model] full fine-tune: {trainable/1e9:.2f}B trainable, "
                  f"{frozen/1e9:.2f}B frozen"
                  + (f" ({', '.join(note)} frozen)" if note else ""))
    elif for_training:
        if cfg.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        targets = find_lora_targets(model, cfg)
        if verbose:
            towers = Counter("vision" if ("visual" in t or "vision" in t) else "language"
                             for t in targets)
            print(f"[model] LoRA on {len(targets)} Linear modules {dict(towers)}")
        lora = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
            use_rslora=cfg.use_rslora,
            use_dora=cfg.use_dora,
        )
        model = get_peft_model(model, lora)

    if for_training:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if verbose and hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    else:
        model.config.use_cache = True
        model.eval()
    return model


def merged_generation_config(model, processor):
    gc_ = model.generation_config
    gc_.pad_token_id = processor.tokenizer.pad_token_id
    eos = processor.tokenizer.eos_token_id
    im_end = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids = [i for i in {eos, im_end} if isinstance(i, int) and i >= 0]
    gc_.eos_token_id = ids if len(ids) > 1 else (ids[0] if ids else eos)
    return gc_


PROCESSOR = load_processor(CFG)
print(f"[model] processor {type(PROCESSOR).__name__} | attn plan: {resolve_attn(CFG)}")

# ### 9.1 Collator smoke test
#
# Builds one real batch and prints exactly which tokens carry loss. If the decoded
# "supervised span" is not precisely the transcription plus the end-of-turn token,
# stop and fix it here — every downstream number depends on this being right.

_smoke_ds = LineDataset(train_df.head(4), CFG, train=True, renderer=None, seed=0)
_smoke_collator = OCRCollator(PROCESSOR, train=True)
_smoke_batch = _smoke_collator([_smoke_ds[i] for i in range(2)])

print("batch keys:", {k: tuple(v.shape) for k, v in _smoke_batch.items() if hasattr(v, "shape")})
_ids = _smoke_batch["input_ids"][0]
_labels = _smoke_batch["labels"][0]
_sup = _ids[_labels != -100]
print(f"\nassistant header repr: {_smoke_collator.assistant_header!r}")
print(f"tokens total {len(_ids)} | supervised {len(_sup)} "
      f"({100*len(_sup)/len(_ids):.1f}%)")
print(f"supervised span -> {PROCESSOR.tokenizer.decode(_sup)!r}")
print(f"ground truth    -> {_smoke_ds.texts[0]!r}")
assert 0 < len(_sup) < len(_ids), "loss mask covers everything or nothing"
assert _smoke_ds.texts[0][:20] in PROCESSOR.tokenizer.decode(_sup), "masking is misaligned"
print("\n[collator] masking verified")
del _smoke_batch, _smoke_ds

# ==========================================================================
# p10_engine.py
# ==========================================================================

# ## 10. Transcription engine
#
# Batched beam search that returns an **N-best list with sequence log-probabilities**,
# not just the top string. Those scores are what makes the fusion in `Section 12`
# possible: rescoring and ROVER both need to know how confident each hypothesis was.
#
# Practical details that matter on a rented GPU:
#
# * left padding for generation (right padding silently corrupts decoder output),
# * width-sorted batching so a 3 200 px line is not padded against a 600 px one,
# * automatic batch-size halving on CUDA OOM instead of losing the run,
# * multi-scale test-time augmentation: the same line is read at several
#   resolutions and the hypotheses are pooled.

def photometric_variant(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return img
    if mode == "dark":     # thicker, higher-contrast ink
        f = np.clip(255.0 * np.power(img / 255.0, 1.35), 0, 255).astype(np.uint8)
        return cv2.erode(f, np.ones((2, 2), np.uint8))
    if mode == "light":    # thinner strokes, flatter paper
        f = np.clip(255.0 * np.power(img / 255.0, 0.78), 0, 255).astype(np.uint8)
        return cv2.dilate(f, np.ones((2, 2), np.uint8))
    if mode == "sharp":
        blur = cv2.GaussianBlur(img, (0, 0), 1.2)
        return np.clip(cv2.addWeighted(img, 1.6, blur, -0.6, 0), 0, 255).astype(np.uint8)
    raise ValueError(mode)


class TranscriptionEngine:
    def __init__(self, model, processor, cfg: Config):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.tokenizer = processor.tokenizer
        self.prompt = processor.apply_chat_template(build_messages(), tokenize=False,
                                                    add_generation_prompt=True)

    # ------------------------------------------------------------------ io --
    def _images(self, ids: Sequence[str], scale: float, mode: str) -> list:
        out = []
        for i in ids:
            img = load_line(i, self.cfg, scale=scale, use_cache=abs(scale - 1.0) < 1e-6)
            if mode != "identity":
                img = photometric_variant(img, mode)
            out.append(to_pil(img))
        return out

    def _clean(self, text: str) -> str:
        text = text.split("<|im_end|>")[0]
        for tag in ("assistant\n", "assistant:", "<|endoftext|>"):
            if text.startswith(tag):
                text = text[len(tag):]
        return normalize_text(text)

    # -------------------------------------------------------------- decode --
    def _greedy_scores(self, out, seqs) -> list:
        """Length-normalised log-probability for a greedy decode."""
        try:
            trans = self.model.compute_transition_scores(
                out.sequences, out.scores, normalize_logits=True)
            pad = self.tokenizer.pad_token_id
            valid = (seqs != pad) if pad is not None else torch.ones_like(seqs, dtype=torch.bool)
            valid = valid[:, :trans.shape[1]]
            trans = torch.nan_to_num(trans[:, :valid.shape[1]], neginf=-20.0, posinf=0.0)
            per_token = (trans * valid).sum(-1) / valid.sum(-1).clamp(min=1)
            return per_token.float().cpu().tolist()
        except Exception:
            return [0.0] * seqs.shape[0]

    @torch.inference_mode()
    def _generate_batch(self, images: list, num_beams: int, n_best: int,
                        length_penalty: float, max_new_tokens: int) -> list:
        side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        try:
            inputs = self.processor(text=[self.prompt] * len(images), images=images,
                                    return_tensors="pt", padding=True)
        finally:
            self.tokenizer.padding_side = side
        inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}
        in_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            # `sequences_scores` is silently dropped unless output_scores is on, and
            # the whole rescoring stage depends on having those scores.
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if num_beams > 1:
            gen_kwargs.update(num_return_sequences=min(n_best, num_beams),
                              length_penalty=length_penalty,
                              early_stopping=True)
        out = self.model.generate(**inputs, **gen_kwargs)

        seqs = out.sequences[:, in_len:]
        texts = self.processor.batch_decode(seqs, skip_special_tokens=True)
        if num_beams > 1 and getattr(out, "sequences_scores", None) is not None:
            scores = out.sequences_scores.float().cpu().tolist()
        else:
            scores = self._greedy_scores(out, seqs)
        del out

        k = min(n_best, num_beams) if num_beams > 1 else 1
        grouped = []
        for b in range(len(images)):
            cand = [(self._clean(texts[b * k + j]), float(scores[b * k + j])) for j in range(k)]
            grouped.append(cand)
        return grouped

    def predict(self, ids: Sequence[str], num_beams: int | None = None,
                n_best: int | None = None, scale: float = 1.0, mode: str = "identity",
                length_penalty: float | None = None, batch_size: int | None = None,
                max_new_tokens: int | None = None, desc: str = "decode",
                leave: bool = True) -> dict:
        cfg = self.cfg
        num_beams = num_beams if num_beams is not None else cfg.num_beams
        n_best = n_best if n_best is not None else cfg.n_best
        length_penalty = length_penalty if length_penalty is not None else cfg.length_penalty
        max_new_tokens = max_new_tokens or cfg.max_new_tokens
        batch_size = batch_size or max(1, cfg.infer_batch_size // max(1, num_beams // 2))

        ids = list(ids)
        widths = []
        for i in ids:
            p = CACHE_DIR / f"{i}.png"
            try:
                with Image.open(p) as im:
                    widths.append(im.size[0])
            except Exception:
                widths.append(10 ** 6)
        order = np.argsort(-np.asarray(widths))

        results: dict[str, list] = {}
        pbar = progress(total=len(ids), desc=desc, unit="img", leave=leave)
        pos = 0
        bs = batch_size
        while pos < len(order):
            chunk = [ids[j] for j in order[pos:pos + bs]]
            try:
                images = self._images(chunk, scale, mode)
                grouped = self._generate_batch(images, num_beams, n_best,
                                               length_penalty, max_new_tokens)
            except torch.cuda.OutOfMemoryError:
                free_gpu()
                if bs == 1:
                    print(f"!! OOM at batch size 1 on {chunk[0]} - emitting empty string")
                    results[chunk[0]] = [("", -50.0)]
                    pos += 1
                    pbar.update(1)
                    continue
                bs = max(1, bs // 2)
                print(f"!! CUDA OOM - retrying with batch size {bs}")
                pbar.set_postfix_str(f"bs={bs}", refresh=False)
                continue
            for image_id, cand in zip(chunk, grouped):
                results[image_id] = cand
            pos += len(chunk)
            pbar.update(len(chunk))
        pbar.close()
        return {i: results.get(i, [("", -50.0)]) for i in ids}

    def predict_tta(self, ids: Sequence[str], scales: Sequence[float] | None = None,
                    photometric: bool | None = None, **kw) -> dict:
        """Pool N-best lists across resolutions and photometric variants."""
        cfg = self.cfg
        scales = list(scales if scales is not None else cfg.tta_scales)
        photometric = cfg.tta_photometric if photometric is None else photometric
        views = [(s, "identity") for s in scales]
        if photometric:
            views.append((scales[0], "dark"))
            views.append((scales[0], "light"))

        pooled: dict[str, list] = defaultdict(list)
        for k, (scale, mode) in enumerate(views):
            res = self.predict(ids, scale=scale, mode=mode,
                               desc=f"decode {k+1}/{len(views)} (x{scale:g},{mode})", **kw)
            for image_id, cand in res.items():
                for rank, (text, score) in enumerate(cand):
                    pooled[image_id].append({"text": text, "score": score,
                                             "view": f"{scale:g}:{mode}", "rank": rank})
        return dict(pooled)


def top1(nbest: dict) -> dict:
    """Best hypothesis per ID from a pooled N-best structure."""
    out = {}
    for image_id, cands in nbest.items():
        if not cands:
            out[image_id] = ""
            continue
        if isinstance(cands[0], dict):
            best = max(cands, key=lambda c: (c["score"], -c["rank"]))["text"]
        else:
            best = max(cands, key=lambda c: c[1])[0]
        out[image_id] = best
    return out


def nbest_to_frame(nbest: dict) -> pd.DataFrame:
    rows = []
    for image_id, cands in nbest.items():
        for c in cands:
            if isinstance(c, dict):
                rows.append({"ID": image_id, **c})
            else:
                rows.append({"ID": image_id, "text": c[0], "score": c[1],
                             "view": "1:identity", "rank": 0})
    return pd.DataFrame(rows)

# ==========================================================================
# p11_train.py
# ==========================================================================

# ## 11. Training
#
# ### Schedule
#
# **Cosine annealing with a warm-up and a non-zero floor.** With only ~3 300 lines per
# fold the run is short (a few hundred optimiser steps), so the shape of the schedule
# is not a detail: a 4% linear warm-up keeps the freshly initialised LoRA `B` matrices
# from destabilising the vision tower, the cosine body spends most of its budget at a
# high learning rate, and the `min_lr_rate` floor (5% of peak) stops the tail of the
# run from becoming a no-op. Combined with `max_grad_norm = 0.5` this converges in
# noticeably fewer epochs than a constant or linear-decay schedule.
#
# ### Other ingredients
#
# * **NEFTune** — uniform noise on the input embeddings during training only. Cheap,
#   and consistently worth a few tenths of a CER point on small SFT sets.
# * **EMA of the adapter weights** — an exponential moving average of the LoRA
#   parameters is kept alongside the live ones and evaluated separately. On short
#   noisy runs the average is almost always better than any individual step.
# * **Generation-based validation** — validation loss is a poor proxy for CER, so the
#   callback actually decodes a held-out subset and scores it with the competition
#   metric. Checkpoint selection uses that score, never the loss.
#
# ### 11.1 Choosing LoRA or a full fine-tune
#
# `CFG.finetune_mode` switches between them and re-derives everything that has to
# change with it, but the two are not interchangeable on this dataset:
#
# | | `"lora"` | `"full"` |
# |---|---|---|
# | Trainable | ~100M (0.5–2%) | every weight |
# | VRAM, 8B backbone | ~28 GB | ~75 GB even with 8-bit Adam |
# | Learning rate | `1.2e-4` | `1e-5` — a LoRA rate destroys the language prior |
# | Checkpoint | ~400 MB | ~16 GB, so hub pushes get slow |
# | EMA | on | off; the fp32 shadow would be model-sized |
# | Overfits 3 300 lines | slowly | quickly — hence 3 epochs, not 5 |
#
# **LoRA is the default because it is the better bet here, not merely the cheaper
# one.** With roughly 3 300 training lines per fold, full fine-tuning has far more
# capacity than the data can constrain, and the regularisation that LoRA imposes by
# construction is doing useful work. A full fine-tune becomes interesting in exactly
# one situation: a smaller backbone (`Qwen3-VL-4B-Instruct` or `-2B`) where the whole
# network fits comfortably and the extra capacity of the 8B model is not available to
# begin with. If you try it, compare out-of-fold scores rather than trusting either
# prior — and note that the two modes make *different* errors, so a LoRA fold and a
# full-fine-tune fold fuse well together in the ROVER vote.
#
# The middle option, and the one worth reaching for first if LoRA looks
# capacity-limited, is not full fine-tuning but a larger rank: `lora_r = 64` with
# `lora_alpha = 128` roughly doubles adapter capacity for a few GB.

from transformers import Trainer, TrainingArguments, TrainerCallback
from transformers.trainer_pt_utils import LengthGroupedSampler


class AdapterEMA(TrainerCallback):
    """Exponential moving average of the trainable (LoRA) parameters."""

    def __init__(self, model, decay: float = 0.999, warmup_steps: int = 20):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.model = model
        self.shadow = {n: p.detach().float().clone()
                       for n, p in model.named_parameters() if p.requires_grad}
        self.backup: dict[str, torch.Tensor] = {}
        self.n_updates = 0
        n_params = sum(v.numel() for v in self.shadow.values())
        print(f"[ema] tracking {len(self.shadow)} tensors / {n_params/1e6:.1f}M params "
              f"({4*n_params/1e9:.2f} GB fp32)")

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        self.n_updates += 1
        # Ramp the decay in so the average is not dominated by the random init.
        d = min(self.decay, (1 + self.n_updates) / (self.warmup_steps + self.n_updates))
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                self.shadow[n].mul_(d).add_(p.detach().float(), alpha=1 - d)

    @torch.no_grad()
    def apply(self) -> None:
        self.backup = {}
        for n, p in self.model.named_parameters():
            if n in self.shadow:
                self.backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n].to(p.dtype))

    @torch.no_grad()
    def restore(self) -> None:
        for n, p in self.model.named_parameters():
            if n in self.backup:
                p.data.copy_(self.backup[n])
        self.backup = {}


class HeartbeatCallback(TrainerCallback):
    """One progress line per logging interval, with a projected finish time.

    The Trainer's own bar is disabled on a headless run - a carriage-return bar
    redrawing ten times a second turns a log file into megabytes of noise - so this
    prints the same information in a form that survives being written to disk.
    """

    def __init__(self, total_steps: int, tag: str, tb: "TBLogger | None" = None):
        self.total = max(int(total_steps), 1)
        self.tag = tag
        self.tb = tb
        self.t0 = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        if "loss" not in logs and "reward" not in logs:
            return
        step = int(state.global_step)
        elapsed = time.time() - self.t0
        rate = step / max(elapsed, 1e-6)
        eta = (self.total - step) / rate if rate > 0 else 0.0
        lr = logs.get("learning_rate", float("nan"))
        extra = "".join(f" {k} {logs[k]:.4f}" for k in ("loss", "reward", "kl")
                        if isinstance(logs.get(k), (int, float)))
        print(f"[{self.tag}] step {step}/{self.total} ({100*step/self.total:5.1f}%)"
              f"{extra} lr {lr:.2e} "
              f"| {fmt_hms(elapsed)} elapsed, ~{fmt_hms(eta)} left "
              f"| epoch {logs.get('epoch', 0):.2f}")
        if self.tb is not None:
            self.tb.scalar("train/eta_minutes", eta / 60, step)


class EpochSeedCallback(TrainerCallback):
    """Re-seeds on-the-fly augmentation and StackMix rendering every epoch."""

    def __init__(self, dataset):
        self.dataset = dataset

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.dataset.set_epoch(int(state.epoch or 0))


class GenerationEvalCallback(TrainerCallback):
    """Decode a validation subset and select checkpoints on the competition metric."""

    def __init__(self, model, processor, cfg: Config, val_df: pd.DataFrame,
                 every_n_steps: int, out_dir: Path, ema: AdapterEMA | None = None,
                 max_samples: int = 320, tag: str = "fold0", tb: "TBLogger | None" = None):
        self.model = model
        self.processor = processor
        self.cfg = cfg
        self.every = max(int(every_n_steps), 1)
        self.out_dir = Path(out_dir)
        self.ema = ema
        self.tag = tag
        self.tb = tb
        sub = val_df if len(val_df) <= max_samples else val_df.sample(max_samples, random_state=0)
        self.ids = sub["ID"].tolist()
        self.refs = sub["Target"].tolist()
        self.history: list[dict] = []
        self.best = {"score": -1.0, "path": None, "step": -1, "source": None}

    def _score(self, engine: TranscriptionEngine) -> dict:
        preds = engine.predict(self.ids, num_beams=1, n_best=1, desc="val",
                               batch_size=None, leave=False)
        hyps = [preds[i][0][0] for i in self.ids]
        return competition_score(self.refs, hyps)

    def _save(self, step: int, source: str) -> Path:
        path = self.out_dir / f"{self.tag}_{source}_step{step}"
        if path.exists():
            shutil.rmtree(path)
        self.model.save_pretrained(str(path))
        if self.cfg.finetune_mode == "full":
            # Full checkpoints are reloaded as standalone models, so they need the
            # processor next to the weights to be usable outside this notebook.
            self.processor.save_pretrained(str(path))
        return path

    def evaluate_now(self, state) -> None:
        step = int(state.global_step)
        was_training = self.model.training
        self.model.eval()
        use_cache = self.model.config.use_cache
        self.model.config.use_cache = True
        try:
            self.model.gradient_checkpointing_disable()
        except Exception:
            pass

        engine = TranscriptionEngine(self.model, self.processor, self.cfg)
        rows = {"step": step}
        candidates = [("raw", self._score(engine))]
        if self.ema is not None:
            self.ema.apply()
            candidates.append(("ema", self._score(engine)))
            self.ema.restore()

        for source, res in candidates:
            rows[f"{source}_score"] = res["score"]
            rows[f"{source}_cer"] = res["cer"]
            rows[f"{source}_wer"] = res["wer"]
            if self.tb is not None:
                self.tb.scalar(f"val/{source}_score", res["score"], step)
                self.tb.scalar(f"val/{source}_cer", res["cer"], step)
                self.tb.scalar(f"val/{source}_wer", res["wer"], step)
        self.history.append(rows)
        if self.tb is not None:
            self.tb.flush()
        msg = " | ".join(f"{s}: score {r['score']:.5f} CER {r['cer']:.4f} WER {r['wer']:.4f}"
                         for s, r in candidates)
        print(f"\n[val step {step}] {msg}")

        best_source, best_res = max(candidates, key=lambda kv: kv[1]["score"])
        if best_res["score"] > self.best["score"]:
            if best_source == "ema":
                self.ema.apply()
            path = self._save(step, best_source)
            if best_source == "ema":
                self.ema.restore()
            if self.best["path"] is not None and Path(self.best["path"]).exists():
                shutil.rmtree(self.best["path"], ignore_errors=True)
            self.best = {"score": best_res["score"], "path": str(path),
                         "step": step, "source": best_source}
            print(f"[val step {step}] new best ({best_source}) -> {path.name}")
            if self.tb is not None:
                self.tb.scalar("val/best_score", best_res["score"], step)

        try:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        except Exception:
            pass
        self.model.config.use_cache = use_cache
        if was_training:
            self.model.train()
        free_gpu()

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step % self.every == 0:
            self.evaluate_now(state)

    def on_train_end(self, args, state, control, **kwargs):
        if not self.history or self.history[-1]["step"] != int(state.global_step):
            self.evaluate_now(state)


class HTRTrainer(Trainer):
    """Trainer with explicit length-grouped batching for a plain torch Dataset."""

    def _get_train_sampler(self, *args, **kwargs):
        if not self.args.group_by_length:
            return super()._get_train_sampler(*args, **kwargs)
        lengths = getattr(self.train_dataset, "lengths", None)
        if lengths is None:
            return super()._get_train_sampler(*args, **kwargs)
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            lengths=lengths, generator=generator,
        )


def make_training_args(cfg: Config, out_dir: Path, steps_per_epoch: int,
                       tb_dir: Path | None = None) -> TrainingArguments:
    kwargs = dict(
        output_dir=str(out_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.lr,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        optim=cfg.optim,
        bf16=True,
        tf32=True,
        logging_steps=max(1, steps_per_epoch // 20),
        eval_strategy="no",
        save_strategy="no",
        # A carriage-return bar is unreadable once redirected to a log file, so on a
        # headless run HeartbeatCallback prints the same information instead.
        disable_tqdm=not INTERACTIVE,
        # Share the log directory with our own writer so loss, learning rate and the
        # generation metrics land on the same TensorBoard run.
        report_to=(["tensorboard"] if (cfg.tensorboard and tb_dir) else "none"),
        logging_dir=str(tb_dir) if tb_dir else None,
        remove_unused_columns=False,
        dataloader_num_workers=cfg.dataloader_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=cfg.dataloader_workers > 0,
        group_by_length=cfg.group_by_length,
        label_smoothing_factor=cfg.label_smoothing,
        seed=cfg.seed,
        data_seed=cfg.seed,             # sampler order independent of model init
        full_determinism=False,         # handled in Section 2.2, without the reseeding
        gradient_checkpointing=False,   # enabled directly on the model
        ddp_find_unused_parameters=False,
    )
    if cfg.neftune_alpha and cfg.neftune_alpha > 0:
        kwargs["neftune_noise_alpha"] = cfg.neftune_alpha
    if cfg.lr_scheduler_type == "cosine" and cfg.min_lr_ratio > 0:
        kwargs["lr_scheduler_type"] = "cosine_with_min_lr"
        kwargs["lr_scheduler_kwargs"] = {"min_lr_rate": cfg.min_lr_ratio}
    try:
        return TrainingArguments(**kwargs)
    except TypeError as exc:   # older transformers: drop the unsupported extras
        print(f"[train] TrainingArguments fallback ({exc})")
        for key in ("lr_scheduler_kwargs", "dataloader_persistent_workers", "tf32",
                    "data_seed", "full_determinism"):
            kwargs.pop(key, None)
        if kwargs.get("lr_scheduler_type") == "cosine_with_min_lr":
            kwargs["lr_scheduler_type"] = "cosine"
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy", "no")
        return TrainingArguments(**kwargs)


def train_fold(fold: int, cfg: Config = None, tag: str | None = None,
               train_override: pd.DataFrame | None = None,
               val_override: pd.DataFrame | None = None) -> dict:
    cfg = cfg or CFG
    tag = tag or f"fold{fold}_seed{cfg.seed}"
    tr_df, va_df = fold_frames(fold)
    if train_override is not None:
        tr_df = train_override
    if val_override is not None:
        va_df = val_override

    set_seed(cfg.seed, cfg.deterministic)
    free_gpu()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    dataset = LineDataset(tr_df, cfg, train=True,
                          renderer=STACKMIX if cfg.stackmix else None, seed=cfg.seed)
    collator = OCRCollator(PROCESSOR, train=True)
    steps_per_epoch = max(1, len(dataset) // (cfg.per_device_batch_size * cfg.grad_accum * max(N_GPU, 1)))
    total_steps = int(steps_per_epoch * cfg.epochs)
    eval_every = max(10, int(steps_per_epoch / max(cfg.evals_per_epoch, 0.1)))

    print(f"\n{'='*78}\n[train] {tag}")
    print(f"[train] real {len(tr_df)} + synthetic {dataset.n_synth} = {len(dataset)} samples")
    print(f"[train] val {len(va_df)} (scoring {min(len(va_df), cfg.eval_max_samples)})")
    print(f"[train] {steps_per_epoch} steps/epoch x {cfg.epochs} epochs = {total_steps} steps"
          f" | eval every {eval_every} | effective batch {EFFECTIVE_BATCH}")
    print(f"[train] mode {cfg.finetune_mode} | seed {cfg.seed} | "
          f"determinism {'on' if cfg.deterministic else 'off'}")
    print(f"[train] cosine annealing: warmup {cfg.warmup_ratio:.0%}, floor {cfg.min_lr_ratio:.0%}"
          f" of peak lr {cfg.lr:g}\n{'='*78}")

    tb = TBLogger(tag, enabled=cfg.tensorboard)
    tb.text("config", "```json\n" + json.dumps(
        {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        indent=2) + "\n```")

    model = build_model(cfg, for_training=True)
    args = make_training_args(cfg, OUT / "checkpoints" / tag, steps_per_epoch, tb.dir)

    ema = AdapterEMA(model, cfg.ema_decay) if cfg.ema_decay and cfg.ema_decay > 0 else None
    val_cb = GenerationEvalCallback(model, PROCESSOR, cfg, va_df, eval_every,
                                    OUT / "checkpoints", ema=ema,
                                    max_samples=cfg.eval_max_samples, tag=tag, tb=tb)
    callbacks = [EpochSeedCallback(dataset), val_cb,
                 HeartbeatCallback(total_steps, tag, tb)]
    if ema is not None:
        callbacks.insert(0, ema)

    trainer = HTRTrainer(model=model, args=args, train_dataset=dataset,
                         data_collator=collator, callbacks=callbacks)
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60
    vram_report(tag)

    hist = pd.DataFrame(val_cb.history)
    hist.to_csv(OUT / f"val_history_{tag}.csv", index=False)
    print(f"\n[train] {tag} finished in {minutes:.1f} min | "
          f"best {val_cb.best['source']} score {val_cb.best['score']:.5f} "
          f"at step {val_cb.best['step']}")

    del trainer, model, ema
    free_gpu()

    tb.hparams(cfg, {"best_score": val_cb.best["score"]})
    tb.close()

    # Upload before anything else can go wrong: a fold that is not on the hub is a
    # fold that dies with the pod.
    pushed = False
    if cfg.hf_push_checkpoints and val_cb.best["path"]:
        pushed = HUB.push_folder(Path(val_cb.best["path"]), f"checkpoints/{tag}",
                                 f"{tag}: val {val_cb.best['score']:.5f}")
        HUB.push_file(OUT / f"val_history_{tag}.csv", f"history/val_history_{tag}.csv")
    push_tensorboard(tag, f"{tag}: metrics")
    NOTIFY.checkpoint(tag, val_cb.best, minutes, pushed)

    return {"tag": tag, "fold": fold, "best": val_cb.best, "history": hist,
            "minutes": minutes, "val_ids": va_df["ID"].tolist(),
            "finetune_mode": cfg.finetune_mode, "seed": cfg.seed}

# ==========================================================================
# p12_run.py
# ==========================================================================

# ## 12. Run training
#
# `CFG.train_folds` controls the budget. Guidance for a single 48 GB card:
#
# | Setting | Wall clock | Use for |
# |---|---|---|
# | `train_folds = (0,)` | ~45 min | first end-to-end pass, sanity check |
# | `train_folds = (0, 1, 2)` | ~2.5 h | ensembling starts paying off here |
# | `train_folds = (0, 1, 2, 3, 4)` | ~4 h | full out-of-fold coverage for tuning |
#
# Each fold writes only its best adapter (selected on generation CER/WER, not loss),
# so disk stays bounded regardless of how many evaluations run. Each fold is also
# pushed to the hub the moment it finishes, so an interrupted run loses at most the
# fold that was in flight.
#
# With `CFG.hf_resume = True` this cell downloads a previous run's checkpoints
# instead of training and jumps straight to decoding — the same weights, therefore
# the same score, without repeating the GPU hours.

RUN_TRAINING = True     # set False to reuse artifacts/runs.json from a previous pass
RUNS: list[dict] = []


def restore_runs_from_hub() -> list:
    """Pull a previous run's checkpoints back down so decoding can start immediately."""
    local = OUT / "hub_restore"
    root = HUB.pull(["runs.json", "checkpoints/**", "history/**"], local)
    if root is None:
        print("[hub] nothing restored - falling back to local artefacts")
        return []
    with open(Path(root) / "runs.json") as fh:
        runs = json.load(fh)
    for r in runs:
        tag = r["tag"]
        # Rewrite the recorded paths onto whatever this machine calls them.
        for key, remote in (("best", f"checkpoints/{tag}"),
                            ("grpo", f"checkpoints/grpo_fold{r['fold']}")):
            path = Path(root) / remote
            if not path.exists():
                continue
            if key == "best":
                r["best"]["path"] = str(path)
            else:
                r["grpo_path"] = str(path)
        hist = Path(root) / "history" / f"val_history_{tag}.csv"
        r["history"] = pd.read_csv(hist) if hist.exists() else pd.DataFrame()
    print(f"[hub] restored {len(runs)} runs from {HUB.repo_id}")
    return runs


if CFG.hf_resume:
    RUNS = restore_runs_from_hub()
    RUN_TRAINING = not RUNS      # only train if the restore came back empty

if RUN_TRAINING:
    stage(f"training {len(CFG.train_folds)} fold(s), mode {CFG.finetune_mode}")
    NOTIFY.run_started()
    _folds = progress(CFG.train_folds, desc="folds", unit="fold")
    for fold in _folds:
        RUNS.append(train_fold(fold))
        _folds.set_postfix_str(f"best {RUNS[-1]['best']['score']:.5f}", refresh=False)
    _folds.close()
    with open(OUT / "runs.json", "w") as fh:
        json.dump([{k: v for k, v in r.items() if k != "history"} for r in RUNS], fh, indent=2)
    HUB.push_file(OUT / "runs.json", "runs.json", "record fold results")
elif not RUNS:
    with open(OUT / "runs.json") as fh:
        RUNS = json.load(fh)
    print(f"[runs] restored {len(RUNS)} runs from disk")

for r in RUNS:
    print(f"  {r['tag']:<24} best {r['best']['score']:.5f} ({r['best']['source']}) "
          f"-> {Path(r['best']['path']).name}")

# Learning curves: check that the cosine tail is flat (converged) rather than still
# improving (under-trained) or turning over (over-fitting).
if RUNS and isinstance(RUNS[0].get("history"), pd.DataFrame) and len(RUNS[0]["history"]):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for r in RUNS:
        h = r["history"]
        ax[0].plot(h["step"], h["raw_score"], marker="o", label=f"{r['tag']} raw")
        if "ema_score" in h:
            ax[0].plot(h["step"], h["ema_score"], marker="s", ls="--", label=f"{r['tag']} ema")
        ax[1].plot(h["step"], h["raw_cer"], marker="o", label=f"{r['tag']} CER")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("competition score"); ax[0].legend(fontsize=7)
    ax[0].set_title("validation score")
    ax[1].set_xlabel("step"); ax[1].set_ylabel("weighted CER"); ax[1].legend(fontsize=7)
    ax[1].set_title("validation CER")
    plt.tight_layout()
    plt.savefig(OUT / "training_curves.png", dpi=110)
    plt.close(fig)
    print(f"[train] curves -> {OUT/'training_curves.png'}")

# ## 13. GRPO polish
#
# Supervised fine-tuning optimises token likelihood; the leaderboard scores edit
# distance. Those are not the same objective, and the gap between them is exactly
# where the last few thousandths of score live. Group Relative Policy Optimisation
# closes it by sampling `G` transcriptions per image, scoring each one with the
# **competition metric as the reward**, and pushing probability mass towards the
# candidates that beat their group's average.
#
# This stage is on by default, but it is the highest-variance part of the notebook:
# it needs a working TRL vision-language `GRPOTrainer`, every step samples `G`
# completions so it is slow, and a bad reward scale can undo the SFT gains. Three
# guards make it safe to leave enabled:
#
# * **A KL leash** (`beta = 0.02`) back to the SFT policy, so it polishes rather than
#   wanders.
# * **A tiny learning rate** (`2e-6`), two orders below the SFT rate.
# * **An acceptance test.** The polished adapter is decoded on held-out data from the
#   same fold and compared with the SFT adapter that produced it. If it does not win,
#   it is discarded and the SFT adapter continues to the decoding stage. Turning GRPO
#   on therefore cannot make the submission worse — only slower.
#
# The gate is the important part. Competitors reported real gains from GRPO on this
# leaderboard, but an ungated RL stage is precisely the kind of thing that silently
# costs a place, and the reward here is a *proxy* computed on training-fold images.

def grpo_batch_shape(cfg: Config) -> tuple[int, int]:
    """TRL requires the effective batch to be a whole number of generation groups."""
    g = max(2, cfg.grpo_num_generations)
    per_device = max(1, min(cfg.per_device_batch_size // 2, g))
    while g % per_device != 0:            # groups must not be split across devices
        per_device -= 1
    accum = max(1, int(round(cfg.grad_accum * cfg.per_device_batch_size / (per_device * g))))
    return per_device, accum


def run_grpo(adapter_path: str | Path, fold: int, cfg: Config = None,
             steps: int | None = None) -> str | None:
    cfg = cfg or CFG
    steps = steps or cfg.grpo_steps
    try:
        from trl import GRPOConfig, GRPOTrainer
    except Exception as exc:
        print(f"[grpo] TRL unavailable ({exc}) - skipping")
        return None

    tr_df, va_df = fold_frames(fold)
    set_seed(cfg.seed + 1, cfg.deterministic)
    free_gpu()

    model = build_model(cfg, adapter_path=adapter_path, for_training=True)
    prompt_text = PROCESSOR.apply_chat_template(build_messages(), tokenize=False,
                                                add_generation_prompt=True)

    from datasets import Dataset as HFDataset

    records = [{"image_id": i, "prompt": prompt_text, "target": t}
               for i, t in zip(tr_df.ID, tr_df.Target)]
    ds = HFDataset.from_list(records)

    def attach_images(batch):
        batch["image"] = [to_pil(load_line(i, cfg)) for i in batch["image_id"]]
        return batch

    ds = ds.with_transform(attach_images)   # lazy: 3k decoded images will not fit in RAM

    def reward_metric(completions, target, **kwargs) -> list:
        """Competition score of each sampled transcription, centred on [-1, 1]."""
        out = []
        for comp, ref in zip(completions, target):
            hyp = normalize_text(comp if isinstance(comp, str) else comp[0].get("content", ""))
            s = competition_score([ref], [hyp])["score"]
            out.append(2.0 * s - 1.0)
        return out

    per_device, accum = grpo_batch_shape(cfg)
    tb_dir = TBLogger.root / f"grpo_fold{fold}"
    print(f"[grpo] fold {fold}: {steps} steps x {cfg.grpo_num_generations} samples/image "
          f"| batch {per_device} x accum {accum} | lr {cfg.grpo_lr:g} beta {cfg.grpo_beta}")
    args = GRPOConfig(
        output_dir=str(OUT / "checkpoints" / f"grpo_fold{fold}"),
        learning_rate=cfg.grpo_lr,
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        num_generations=cfg.grpo_num_generations,
        max_completion_length=cfg.max_new_tokens,
        max_steps=steps,
        temperature=cfg.grpo_temperature,
        beta=cfg.grpo_beta,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=(["tensorboard"] if cfg.tensorboard else "none"),
        logging_dir=str(tb_dir),
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        seed=cfg.seed + 1,
        remove_unused_columns=False,
        disable_tqdm=not INTERACTIVE,
    )
    trainer = GRPOTrainer(model=model, args=args, reward_funcs=[reward_metric],
                          train_dataset=ds, processing_class=PROCESSOR,
                          callbacks=[HeartbeatCallback(steps, f"grpo fold{fold}")])
    trainer.train()

    out_path = OUT / "checkpoints" / f"grpo_fold{fold}_final"
    model.save_pretrained(str(out_path))
    if cfg.finetune_mode == "full":
        PROCESSOR.save_pretrained(str(out_path))
    del trainer, model
    free_gpu()
    print(f"[grpo] weights -> {out_path}")
    return str(out_path)


def score_checkpoint(path: str | Path, ids: list, refs: list, cfg: Config = None) -> float:
    """Decode a held-out subset with one checkpoint and return the competition score."""
    cfg = cfg or CFG
    free_gpu()
    model = build_model(cfg, adapter_path=path, for_training=False, verbose=False)
    merged_generation_config(model, PROCESSOR)
    engine = TranscriptionEngine(model, PROCESSOR, cfg)
    preds = engine.predict(ids, num_beams=1, n_best=1, desc="grpo-eval", batch_size=None)
    score = competition_score(refs, [preds[i][0][0] for i in ids])["score"]
    del engine, model
    free_gpu()
    return score


def grpo_stage(run: dict, cfg: Config = None) -> dict:
    """Polish one fold with GRPO and keep the result only if it beats its parent."""
    cfg = cfg or CFG
    fold = run["fold"]
    sft_path = run["best"]["path"]
    grpo_path = run_grpo(sft_path, fold, cfg)
    if not grpo_path:
        return run

    if not cfg.grpo_keep_if_better:
        run["grpo_path"] = grpo_path
        print("[grpo] acceptance test skipped (grpo_keep_if_better = False)")
        return run

    _, va_df = fold_frames(fold)
    sub = va_df if len(va_df) <= cfg.grpo_eval_samples else va_df.sample(
        cfg.grpo_eval_samples, random_state=cfg.seed)
    ids, refs = sub["ID"].tolist(), sub["Target"].tolist()
    sft_score = score_checkpoint(sft_path, ids, refs, cfg)
    grpo_score = score_checkpoint(grpo_path, ids, refs, cfg)
    delta = grpo_score - sft_score
    run["grpo_scores"] = {"sft": sft_score, "grpo": grpo_score, "delta": delta}
    print(f"[grpo] fold {fold}: SFT {sft_score:.5f} vs GRPO {grpo_score:.5f} ({delta:+.5f})")

    tb = TBLogger(f"grpo_fold{fold}", enabled=cfg.tensorboard)
    tb.scalars({"sft": sft_score, "grpo": grpo_score, "delta": delta}, 0, prefix="verdict/")
    tb.close()

    accepted = delta > 0
    if accepted:
        run["grpo_path"] = grpo_path
        print(f"[grpo] fold {fold}: ACCEPTED, decoding will use the polished weights")
        if cfg.hf_push_checkpoints:
            HUB.push_folder(Path(grpo_path), f"checkpoints/grpo_fold{fold}",
                            f"grpo fold{fold}: {grpo_score:.5f} ({delta:+.5f})")
    else:
        print(f"[grpo] fold {fold}: REJECTED, keeping the SFT adapter")
    push_tensorboard(f"grpo_fold{fold}", f"grpo fold{fold} metrics")
    NOTIFY.grpo(fold, run["grpo_scores"], accepted)
    return run


if CFG.run_grpo and RUNS:
    stage("GRPO polish + acceptance test")
    RUNS = [grpo_stage(r) for r in progress(RUNS, desc="grpo", unit="fold")]
    _kept = sum("grpo_path" in r for r in RUNS)
    print(f"\n[grpo] {_kept}/{len(RUNS)} folds improved and will decode with GRPO weights")
else:
    print("[grpo] disabled (CFG.run_grpo = False)")

# ==========================================================================
# p13_decode.py
# ==========================================================================

# ## 14. Out-of-fold and test decoding
#
# Each trained adapter is loaded **once** and used to decode both its own validation
# fold and the full test set, so the base model is materialised as few times as
# possible. Every decode produces an N-best list with beam scores across all TTA
# views; those pooled lists are the input to the fusion stage.
#
# Cost control: the number of forward passes is
# `folds x (len(scales) + 2 if photometric) x beams`. On a 48 GB card, one adapter
# over the 1 374 test images with 4 views and 5 beams takes roughly an hour. Set
# `FAST_DECODE = True` for a single greedy view while iterating.

FAST_DECODE = False

DECODE_KW = dict(num_beams=1, n_best=1) if FAST_DECODE else dict(
    num_beams=CFG.num_beams, n_best=CFG.n_best)
DECODE_SCALES = (1.0,) if FAST_DECODE else CFG.tta_scales
DECODE_PHOTOMETRIC = False if FAST_DECODE else CFG.tta_photometric


def save_nbest(nbest: dict, path: Path) -> None:
    nbest_to_frame(nbest).to_csv(path, index=False)


def load_nbest(path: Path) -> dict:
    df = pd.read_csv(path)
    df["text"] = df["text"].fillna("")
    out: dict[str, list] = defaultdict(list)
    for row in df.itertuples():
        out[row.ID].append({"text": row.text, "score": float(row.score),
                            "view": row.view, "rank": int(row.rank)})
    return dict(out)


def decode_run(run: dict, cfg: Config = None, force: bool = False) -> dict:
    cfg = cfg or CFG
    tag = run["tag"]
    adapter = run.get("grpo_path") or run["best"]["path"]
    oof_path = OUT / "preds" / f"oof_{tag}.csv"
    test_path = OUT / "preds" / f"test_{tag}.csv"

    if oof_path.exists() and test_path.exists() and not force:
        print(f"[decode] {tag}: reusing cached predictions")
        return {"tag": tag, "oof": load_nbest(oof_path), "test": load_nbest(test_path)}

    free_gpu()
    model = build_model(cfg, adapter_path=adapter, for_training=False)
    merged_generation_config(model, PROCESSOR)
    engine = TranscriptionEngine(model, PROCESSOR, cfg)

    val_ids = run["val_ids"]
    print(f"[decode] {tag}: {len(val_ids)} oof + {len(test_df)} test images")
    oof = engine.predict_tta(val_ids, scales=DECODE_SCALES,
                             photometric=DECODE_PHOTOMETRIC, **DECODE_KW)
    test = engine.predict_tta(test_df.ID.tolist(), scales=DECODE_SCALES,
                              photometric=DECODE_PHOTOMETRIC, **DECODE_KW)
    save_nbest(oof, oof_path)
    save_nbest(test, test_path)

    del engine, model
    free_gpu()

    refs = train_df.set_index("ID").loc[val_ids, "Target"].tolist()
    hyps = [top1(oof)[i] for i in val_ids]
    res = competition_score(refs, hyps)
    print(f"[decode] {tag}: raw top-1 OOF score {res['score']:.5f} "
          f"(CER {res['cer']:.4f}, WER {res['wer']:.4f})")
    return {"tag": tag, "oof": oof, "test": test}


stage("decode out-of-fold + test")
DECODES = [decode_run(r) for r in progress(RUNS, desc="adapters", unit="model")]

# Assemble the out-of-fold table. With every fold trained this covers the whole
# training set; with a subset it covers those folds only - still enough to tune the
# post-processing, just noisier.
OOF_NBEST: dict[str, list] = defaultdict(list)
for d in DECODES:
    for image_id, cands in d["oof"].items():
        for c in cands:
            OOF_NBEST[image_id].append({**c, "model": d["tag"]})
OOF_NBEST = dict(OOF_NBEST)

TEST_NBEST: dict[str, list] = defaultdict(list)
for d in DECODES:
    for image_id, cands in d["test"].items():
        for c in cands:
            TEST_NBEST[image_id].append({**c, "model": d["tag"]})
TEST_NBEST = dict(TEST_NBEST)

OOF_IDS = [i for i in train_df.ID if i in OOF_NBEST]
OOF_REFS = train_df.set_index("ID").loc[OOF_IDS, "Target"].tolist()
print(f"[oof] {len(OOF_IDS)} lines covered "
      f"({len(OOF_IDS)/len(train_df):.0%} of train), "
      f"{np.mean([len(OOF_NBEST[i]) for i in OOF_IDS]):.1f} hypotheses each")

BASELINE = competition_score(OOF_REFS, [top1(OOF_NBEST)[i] for i in OOF_IDS], verbose=True)
print(f"[oof] baseline (best-beam, no post-processing): {BASELINE['score']:.5f}")

# Where do the errors live? Oracle score = the ceiling that perfect hypothesis
# selection would reach. A large gap means fusion and rescoring have room to work;
# a small gap means the model itself has to get better.
_oracle, _worst = [], []
for image_id, ref in progress(list(zip(OOF_IDS, OOF_REFS)), desc="oracle", unit="line"):
    cands = {c["text"] for c in OOF_NBEST[image_id]}
    scored = [(competition_score([ref], [t])["score"], t) for t in cands]
    _oracle.append(max(scored)[1])
    _worst.append(min(scored)[1])
print(f"[oof] oracle over N-best : {competition_score(OOF_REFS, _oracle)['score']:.5f}")
print(f"[oof] anti-oracle        : {competition_score(OOF_REFS, _worst)['score']:.5f}")
print(f"[oof] headroom from selection alone: "
      f"{competition_score(OOF_REFS, _oracle)['score'] - BASELINE['score']:+.5f}")

# ==========================================================================
# p13a_ctc.py
# ==========================================================================

# ### 14.1 Second recogniser: kraken PP-OCRv6 (CTC)
#
# **Why this exists.** The BHL OCR leaderboard (finebooks/bhl-ocr-leaderboard,
# updated 2026-09-03) found that kraken's 15.9M-parameter PP-OCRv6 port reads
# 18th–19th-century book pages about as well as the best VLMs hundreds of times its
# size: reading-lane CER .034 [.024–.044] against .051 [.029–.077] for Qwen3.5-9B,
# .041 vs .067 diplomatic CER, and **0 % looped pages** against 1.5 %.
#
# Three things temper that for this challenge, and they shape how it is used here:
#
# * The board is **zero-shot on printed books**; this task is **fine-tuned on
#   17th-century handwriting**. A VLM fine-tuned on 3 300 in-domain lines is a
#   different animal from one read zero-shot, so the ranking may not transfer.
# * The reading-CER confidence intervals overlap. The board's own guidance is that
#   overlapping intervals mean "within noise". The differences that are clearly
#   outside noise are the *failure modes*: CTC never loops and never invents text
#   on near-empty lines, because it emits one label per image column.
# * Different architecture means different mistakes, and that is exactly what a
#   ROVER vote converts into accuracy.
#
# So the CTC model is **not a replacement** for the VLM. It is fine-tuned per fold on
# the same lines, decoded with photometric TTA, and handed to Section 15 as
#
# 1. extra hypotheses in the confusion network, and
# 2. an **anchor feature**: how close each VLM hypothesis is to what the CTC model
#    read, which penalises runaway or hallucinated VLM output.
#
# Section 15 tunes the fusion twice — with and without the CTC member — and keeps
# the CTC member only if it wins out-of-fold, the same gate GRPO uses. It also
# reports the zero-shot and fine-tuned CTC scores on *our* data, which is the
# direct test of whether the leaderboard finding carries over.
#
# **Isolation.** kraken 7.1 pins `torch>=2.9`, `scipy~=1.15` and `scikit-learn~=1.7`,
# which would fight the VLM stack. It therefore lives in its own virtualenv
# (`CFG.ctc_venv`) and is driven through subprocesses; nothing here can break the
# main environment. Any failure in this section prints a warning and the pipeline
# continues VLM-only.
#
# **Cost.** A 16M-parameter model on ~3 300 lines: a few minutes per fold on the
# card that just trained an 8B VLM, which is why every fold is trained by default
# (`CFG.ctc_folds = ()`), even when the VLM trains only one.

import dataclasses
import subprocess

import cv2
from PIL import Image

CTC_ROOT = OUT / "ctc"
CTC_LINES = CTC_ROOT / "lines"          # <ID>.png (+ <ID>.gt.txt for train)
CTC_MODELS = CTC_ROOT / "models"
CTC_LOGS = CTC_ROOT / "logs"


def ctc_config(cfg: Config = None) -> Config:
    """Image settings for the CTC branch: same cleaning, kraken's line height."""
    cfg = cfg or CFG
    return dataclasses.replace(cfg, line_height=cfg.ctc_line_height,
                               max_line_width=cfg.ctc_max_width or 8192,
                               max_upscale=max(cfg.max_upscale, 3.0))


def ctc_folds(cfg: Config = None) -> tuple:
    cfg = cfg or CFG
    return tuple(cfg.ctc_folds) if cfg.ctc_folds else tuple(range(cfg.n_folds))


def ctc_line(image_id: str, cfg: Config = None, mode: str = "identity") -> np.ndarray:
    """Background-flattened, deskewed line at the CTC model's input height.

    kraken height-normalises again internally; doing the cleaning here keeps the
    two recognisers looking at the same pixels, so their disagreements are about
    reading rather than preprocessing.
    """
    ccfg = ctc_config(cfg)
    path = image_path(image_id)
    if path is None:
        return np.full((ccfg.line_height, ccfg.min_line_width), 235, np.uint8)
    img = normalize_line(read_gray(path), ccfg)
    if mode != "identity":
        img = photometric_variant(img, mode)
    return img


def write_ctc_lines(ids: Sequence[str], texts: Sequence[str] | None,
                    dest: Path, cfg: Config = None, mode: str = "identity") -> list:
    """Write kraken `path`-format ground truth: X.png next to X.gt.txt."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    texts = list(texts) if texts is not None else [None] * len(ids)
    for image_id, text in zip(ids, texts):
        png = dest / f"{image_id}.png"
        if not png.exists():
            cv2.imwrite(str(png), ctc_line(image_id, cfg, mode))
        if text is not None:
            gt = dest / f"{image_id}.gt.txt"
            if not gt.exists() or gt.read_text(encoding="utf-8") != text:
                gt.write_text(text, encoding="utf-8")
        paths.append(png)
    return paths


def write_manifest(paths: Sequence[Path], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(Path(p).resolve()) for p in paths) + "\n",
                    encoding="utf-8")
    return path


def ctc_width_report(paths: Sequence[Path], cfg: Config = None) -> dict:
    """Widths *after* kraken rescales each line to the model height.

    ketos silently drops training lines wider than --max-width, and the width that
    matters is the one after its own height normalisation, not the PNG's. A crop
    that `normalize_line` had to shrink below 128 px to fit its width cap is scaled
    back up by kraken - and would vanish from training without this check.
    """
    cfg = cfg or CFG
    widths = []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
        widths.append(w * cfg.ctc_line_height / max(h, 1))
    widths = np.asarray(widths) if widths else np.zeros(1)
    # +64: kraken pads 16 px each side before checking, plus augmentation slack.
    auto = int(math.ceil((widths.max() + 64) / 64.0) * 64)
    limit = cfg.ctc_max_width or auto
    return {"n": int(len(widths)), "max": int(widths.max()),
            "p99": float(np.percentile(widths, 99)), "max_width": int(limit),
            "over_limit": int((widths > limit).sum())}


def venv_bin(venv: Path, name: str) -> Path:
    sub = "Scripts" if os.name == "nt" else "bin"
    exe = f"{name}.exe" if os.name == "nt" else name
    return Path(venv) / sub / exe


def ketos_train_cmd(ketos: Path, base_model: Path, train_manifest: Path,
                    val_manifest: Path, out_dir: Path, cfg: Config = None,
                    max_width: int | None = None) -> list:
    """Fine-tune the PP-OCRv6 checkpoint. Every flag below exists in kraken 7.1.1.

    * `--resize union` keeps the pretrained codec and adds the challenge-only
      symbols (`^`, `~`, ...) as new output classes.
    * `-u NFD` matches how the base model was trained; predictions are folded back
      to NFC by `normalize_text` before scoring.
    * `--deterministic` is kraken's deterministic mode, which is `warn`-only for Lightning — the
      CUDA CTC-loss backward has no deterministic kernel and would otherwise abort.
    """
    cfg = cfg or CFG
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = "bf16-mixed" if (torch.cuda.is_available()
                                 and torch.cuda.is_bf16_supported()) else "32-true"
    cmd = [str(ketos), "-v", "-d", device, "--precision", precision,
           "--workers", str(min(8, max(1, cfg.dataloader_workers))),
           "-s", str(cfg.seed)]
    if cfg.deterministic:
        cmd.append("--deterministic")
    cmd += ["train", "-f", "path", "-i", str(base_model), "--resize", "union",
            "-u", "NFD", "--normalize-whitespace",
            "-t", str(train_manifest), "-e", str(val_manifest),
            "-o", str(out_dir), "-B", str(cfg.ctc_batch_size),
            "-N", str(cfg.ctc_epochs), "-q", "fixed", "-F", "1",
            "--lrate", str(cfg.ctc_lr), "--warmup", str(cfg.ctc_warmup),
            "--max-width", str(max_width or cfg.ctc_max_width or 4096)]
    cmd.append("--augment" if cfg.ctc_augment else "--no-augment")
    return cmd


def best_ctc_weights(out_dir: Path) -> Path | None:
    """ketos writes best_<val_metric>.safetensors next to its checkpoints."""
    found = []
    for p in Path(out_dir).glob("best_*.safetensors"):
        try:
            found.append((float(p.stem.split("_", 1)[1]), p))
        except ValueError:
            continue
    return max(found)[1] if found else None


# The recogniser runs inside the kraken venv. Lines are tiled onto one tall canvas
# per batch so kraken sees a multi-line bbox segmentation and batches them through
# the network together, instead of one call per 16M-parameter forward pass.
CTC_WORKER = r'''
import json, math, sys
from pathlib import Path
from PIL import Image
from kraken.containers import BBoxLine, Segmentation
from kraken.configs import RecognitionInferenceConfig
from kraken.tasks import RecognitionTaskModel

job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model = RecognitionTaskModel.load_model(job["model"])
dev = job.get("device", "cpu")
if dev.startswith("cuda"):
    accelerator, devices = "gpu", [int(dev.split(":")[1]) if ":" in dev else 0]
else:
    accelerator, devices = "cpu", 1
config = RecognitionInferenceConfig(batch_size=job["batch"], num_line_workers=0,
                                    accelerator=accelerator, device=devices)
items, gap, out = job["items"], 16, []
for start in range(0, len(items), job["batch"]):
    chunk = items[start:start + job["batch"]]
    ims = [Image.open(it["path"]).convert("L") for it in chunk]
    width = max(im.width for im in ims)
    height = sum(im.height for im in ims) + gap * (len(ims) + 1)
    canvas = Image.new("L", (width, height), 255)
    lines, y = [], gap
    for k, im in enumerate(ims):
        canvas.paste(im, (0, y))
        lines.append(BBoxLine(id=f"l{k}", bbox=(0, y, im.width, y + im.height)))
        y += im.height + gap
    seg = Segmentation(type="bbox", imagename="canvas", text_direction="horizontal-lr",
                       script_detection=False, lines=lines)
    for it, rec in zip(chunk, model.predict(im=canvas, segmentation=seg, config=config)):
        conf = [max(float(c), 1e-6) for c in (rec.confidences or [])]
        score = sum(math.log(c) for c in conf) / len(conf) if conf else -5.0
        out.append({"ID": it["id"], "text": rec.prediction, "score": score})
    print(f"ctc worker {min(start + len(chunk), len(items))}/{len(items)}", flush=True)
Path(job["out"]).write_text(json.dumps(out), encoding="utf-8")
'''


def run_logged(cmd: list, log_path: Path, env: dict | None = None,
               echo_every: float = 30.0) -> int:
    """Run a subprocess, tee its output to a log, echo a line every `echo_every` s."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[ctc] $", " ".join(map(str, cmd))[:400])
    last = 0.0
    with open(log_path, "a", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", env=env)
        for line in proc.stdout:
            fh.write(line)
            now = time.time()
            if INTERACTIVE or now - last > echo_every or "error" in line.lower():
                print(f"[ctc]   {line.rstrip()[:200]}")
                last = now
        return proc.wait()


def ctc_frame_to_nbest(df: pd.DataFrame, model_tag: str) -> dict:
    """Rows of (ID, text, score, view) -> the pooled N-best structure of Section 14."""
    out: dict[str, list] = defaultdict(list)
    for row in df.itertuples():
        out[row.ID].append({"text": normalize_text(row.text), "score": float(row.score),
                            "view": row.view, "rank": 0,
                            "model": model_tag, "family": "ctc"})
    return dict(out)


def merge_nbest(base: dict, extra: dict, only: set | None = None) -> dict:
    merged = {k: list(v) for k, v in base.items()}
    for image_id, cands in extra.items():
        if only is not None and image_id not in only:
            continue
        merged.setdefault(image_id, []).extend(cands)
    return merged

CTC_STATUS = "disabled"
CTC_REPORT: dict = {}
CTC_DECODES: list[dict] = []


def ensure_kraken_env(cfg: Config = None) -> tuple[Path, Path]:
    """Create (once) an isolated venv with kraken; return (python, ketos)."""
    cfg = cfg or CFG
    venv = WORK / cfg.ctc_venv
    py, ketos = venv_bin(venv, "python"), venv_bin(venv, "ketos")
    probe = [str(py), "-c", "import kraken.train.ppocr, kraken.tasks; import kraken; "
                            "print(kraken.__version__ if hasattr(kraken,'__version__') else '')"]
    if py.exists() and subprocess.run(probe, capture_output=True).returncode == 0:
        return py, ketos
    log = CTC_LOGS / "install.log"
    if not py.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    pip_ = [str(py), "-m", "pip", "install", "-q", "--no-cache-dir"]
    run_logged(pip_ + ["--upgrade", "pip"], log)
    if cfg.ctc_torch_index:
        rc = run_logged(pip_ + ["--index-url", cfg.ctc_torch_index,
                                "torch>=2.9,<=2.14", "torchvision"], log)
        if rc != 0:
            print(f"[ctc] torch from {cfg.ctc_torch_index} failed; trying PyPI")
    if run_logged(pip_ + [f"kraken=={cfg.ctc_kraken_version}"], log) != 0:
        raise RuntimeError(f"kraken install failed - see {log}")
    if subprocess.run(probe, capture_output=True).returncode != 0:
        raise RuntimeError(f"kraken imports fail in {venv} - see {log}")
    return py, ketos


def ctc_decode(py: Path, model: Path, ids: Sequence[str], tag: str,
               cfg: Config = None) -> pd.DataFrame:
    """Decode `ids` with one CTC model across the photometric TTA views."""
    cfg = cfg or CFG
    frames = []
    for mode in cfg.ctc_tta_modes:
        line_dir = CTC_LINES if mode == "identity" else CTC_ROOT / f"lines_{mode}"
        paths = write_ctc_lines(ids, None, line_dir, cfg, mode=mode)
        job = CTC_ROOT / "jobs" / f"{tag}_{mode}.json"
        res = CTC_ROOT / "jobs" / f"{tag}_{mode}.out.json"
        job.parent.mkdir(parents=True, exist_ok=True)
        job.write_text(json.dumps({
            "model": str(model), "batch": int(cfg.ctc_infer_batch), "out": str(res),
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "items": [{"id": i, "path": str(p.resolve())} for i, p in zip(ids, paths)],
        }), encoding="utf-8")
        worker = CTC_ROOT / "ctc_worker.py"
        worker.write_text(CTC_WORKER, encoding="utf-8")
        rc = run_logged([str(py), str(worker), str(job)], CTC_LOGS / f"decode_{tag}.log")
        if rc != 0 or not res.exists():
            raise RuntimeError(f"ctc decode {tag}/{mode} failed (rc={rc})")
        df = pd.DataFrame(json.loads(res.read_text(encoding="utf-8")))
        df["view"] = f"ctc:{mode}"
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["text"] = out["text"].fillna("").map(normalize_text)
    return out


def ctc_top1_score(df: pd.DataFrame, ids: Sequence[str]) -> dict:
    ident = df[df.view == "ctc:identity"].set_index("ID")["text"]
    refs = train_df.set_index("ID").loc[list(ids), "Target"].tolist()
    return competition_score(refs, [ident.get(i, "") for i in ids])


def run_ctc_branch(cfg: Config = None) -> list:
    cfg = cfg or CFG
    folds = ctc_folds(cfg)
    stage(f"kraken CTC branch: {len(folds)} fold(s)")
    for d in (CTC_ROOT, CTC_MODELS, CTC_LOGS, CTC_ROOT / "preds"):
        d.mkdir(parents=True, exist_ok=True)

    if cfg.hf_resume and HUB.enabled:
        HUB.pull(["ctc/models/**", "ctc/preds/**"], OUT)

    _env: dict = {}

    def env() -> tuple[Path, Path]:
        """Build the kraken venv only if something actually needs to run in it."""
        if not _env:
            _env["py"], _env["ketos"] = ensure_kraken_env(cfg)
        return _env["py"], _env["ketos"]

    from huggingface_hub import hf_hub_download

    base = Path(hf_hub_download(cfg.ctc_model_id, cfg.ctc_model_file,
                                local_dir=str(CTC_ROOT / "base")))
    print(f"[ctc] base model {cfg.ctc_model_id} -> {base.name} ({CTC_LICENCE})")

    all_paths = write_ctc_lines(train_df.ID.tolist(), train_df.Target.tolist(),
                                CTC_LINES, cfg)
    widths = ctc_width_report(all_paths, cfg)
    print(f"[ctc] {widths['n']} train lines at h={cfg.ctc_line_height}: "
          f"max width {widths['max']}px, p99 {widths['p99']:.0f}px "
          f"-> ketos --max-width {widths['max_width']}")
    if widths["over_limit"]:
        print(f"!! {widths['over_limit']} lines exceed ctc_max_width="
              f"{widths['max_width']} and will be DROPPED by ketos - set "
              f"CFG.ctc_max_width = 0 to size it from the data")
    CTC_REPORT["widths"] = widths

    # Zero-shot check: does the leaderboard result carry over to secretary hand?
    zs_ids = OOF_IDS if OOF_IDS else train_df.ID.tolist()
    zs_path = CTC_ROOT / "preds" / "oof_ctc_zeroshot.csv"
    if zs_path.exists():
        zs = pd.read_csv(zs_path, keep_default_na=False)
    else:
        zs = ctc_decode(env()[0], base, zs_ids, "zeroshot",
                        dataclasses.replace(cfg, ctc_tta_modes=("identity",)))
        zs.to_csv(zs_path, index=False)
        HUB.push_file(zs_path, f"ctc/preds/{zs_path.name}", "ctc zero-shot oof")
    CTC_REPORT["zero_shot"] = ctc_top1_score(zs, zs_ids)
    print(f"[ctc] zero-shot PP-OCRv6 on our lines: "
          f"score {CTC_REPORT['zero_shot']['score']:.5f} "
          f"(CER {CTC_REPORT['zero_shot']['cer']:.4f}, "
          f"WER {CTC_REPORT['zero_shot']['wer']:.4f}) | VLM raw top-1 "
          f"{BASELINE['score']:.5f}")

    decodes = []
    for fold in progress(folds, desc="ctc folds", unit="fold"):
        tag = f"ctc_fold{fold}"
        oof_path = CTC_ROOT / "preds" / f"oof_{tag}.csv"
        test_path = CTC_ROOT / "preds" / f"test_{tag}.csv"
        tr_df, va_df = fold_frames(fold)
        out_dir = CTC_MODELS / tag
        weights = best_ctc_weights(out_dir)

        if cfg.ctc_finetune and weights is None:
            t0 = time.time()
            tr_m = write_manifest([CTC_LINES / f"{i}.png" for i in tr_df.ID],
                                  CTC_ROOT / "manifests" / f"{tag}_train.txt")
            va_m = write_manifest([CTC_LINES / f"{i}.png" for i in va_df.ID],
                                  CTC_ROOT / "manifests" / f"{tag}_val.txt")
            out_dir.mkdir(parents=True, exist_ok=True)
            (CTC_LOGS / f"train_{tag}.log").unlink(missing_ok=True)   # count this run only
            rc = run_logged(ketos_train_cmd(env()[1], base, tr_m, va_m, out_dir, cfg,
                                            max_width=widths["max_width"]),
                            CTC_LOGS / f"train_{tag}.log")
            weights = best_ctc_weights(out_dir)
            _log = (CTC_LOGS / f"train_{tag}.log").read_text(encoding="utf-8", errors="replace")
            _dropped = _log.count("Failed. Replacing")
            if _dropped:
                print(f"!! {tag}: ketos replaced {_dropped} unusable samples (too wide "
                      f"or too many characters for the width) - see the train log")
            CTC_REPORT[f"{tag}_replaced_samples"] = _dropped
            if rc != 0 or weights is None:
                raise RuntimeError(f"ketos train {tag} failed (rc={rc}); "
                                   f"see {CTC_LOGS / f'train_{tag}.log'}")
            print(f"[ctc] {tag}: {weights.name} in {fmt_hms(time.time() - t0)}")
            if cfg.hf_push_checkpoints:
                HUB.push_file(weights, f"ctc/models/{tag}/{weights.name}",
                              f"{tag}: {weights.stem}")
        weights = weights or base

        if oof_path.exists() and test_path.exists():
            oof = pd.read_csv(oof_path, keep_default_na=False)
            test = pd.read_csv(test_path, keep_default_na=False)
        else:
            oof = ctc_decode(env()[0], weights, va_df.ID.tolist(), f"{tag}_oof", cfg)
            test = ctc_decode(env()[0], weights, test_df.ID.tolist(), f"{tag}_test", cfg)
            oof.to_csv(oof_path, index=False)
            test.to_csv(test_path, index=False)
            HUB.push_file(oof_path, f"ctc/preds/{oof_path.name}", f"{tag} oof")
            HUB.push_file(test_path, f"ctc/preds/{test_path.name}", f"{tag} test")
        res = ctc_top1_score(oof, va_df.ID.tolist())
        CTC_REPORT[tag] = res
        print(f"[ctc] {tag}: fine-tuned top-1 OOF score {res['score']:.5f} "
              f"(CER {res['cer']:.4f}, WER {res['wer']:.4f})")
        decodes.append({"tag": tag, "oof": ctc_frame_to_nbest(oof, tag),
                        "test": ctc_frame_to_nbest(test, tag)})
    return decodes


if CFG.ctc_branch:
    try:
        CTC_DECODES = run_ctc_branch()
        CTC_STATUS = f"ok ({len(CTC_DECODES)} fold models)"
    except Exception as exc:
        import traceback

        CTC_DECODES = []
        CTC_STATUS = f"failed: {type(exc).__name__}: {exc}"
        print(f"!! CTC branch {CTC_STATUS}\n{traceback.format_exc()[-2000:]}")
        print("!! continuing VLM-only; the submission is unaffected")
        NOTIFY.send("CTC branch failed - continuing VLM-only",
                    f"{CTC_STATUS}\n\n{traceback.format_exc()[-6000:]}")
print(f"[ctc] status: {CTC_STATUS}")

# Pool the CTC hypotheses with the VLM ones. OOF only for lines the VLM also covers,
# so both families are tuned on exactly the same rows.
_oof_set = set(OOF_IDS)
for _d in CTC_DECODES:
    OOF_NBEST = merge_nbest(OOF_NBEST, _d["oof"], only=_oof_set)
    TEST_NBEST = merge_nbest(TEST_NBEST, _d["test"])
HAS_CTC = any(c.get("family") == "ctc" for i in OOF_IDS for c in OOF_NBEST.get(i, []))

if HAS_CTC:
    _ctc_all = pd.concat([pd.read_csv(CTC_ROOT / "preds" / f"oof_{d['tag']}.csv",
                                      keep_default_na=False) for d in CTC_DECODES])
    CTC_REPORT["oof_top1"] = ctc_top1_score(_ctc_all, OOF_IDS)
    _oracle_ctc = []
    for image_id, ref in zip(OOF_IDS, OOF_REFS):
        cands = {c["text"] for c in OOF_NBEST[image_id]}
        _oracle_ctc.append(max((competition_score([ref], [t])["score"], t) for t in cands)[1])
    CTC_REPORT["oracle_with_ctc"] = competition_score(OOF_REFS, _oracle_ctc)["score"]
    print(f"[ctc] CTC top-1 on the VLM's OOF rows : {CTC_REPORT['oof_top1']['score']:.5f}")
    print(f"[ctc] VLM raw top-1 on the same rows  : {BASELINE['score']:.5f}")
    print(f"[ctc] oracle over VLM+CTC hypotheses  : {CTC_REPORT['oracle_with_ctc']:.5f}")

# ==========================================================================
# p14_fusion.py
# ==========================================================================

# ## 15. Fusion, rescoring and repair
#
# Beam search returns the most likely *token* sequence. The leaderboard rewards the
# lowest *edit distance*, and those are not the same objective. Three mechanisms
# close the gap, all tuned on out-of-fold predictions and all reversible if they do
# not help:
#
# 1. **ROVER** — build a confusion network from every hypothesis (all beams, all TTA
#    views, all folds) and take a weighted vote per aligned slot. Voting fixes the
#    single-character disagreements that dominate CER, and can produce a string that
#    no individual hypothesis proposed.
# 2. **N-best rescoring** — re-rank hypotheses with a linear model over beam score,
#    cross-view agreement, a character n-gram language model, out-of-vocabulary rate
#    and length prior. Weights come from random search against the real metric.
# 3. **Lexicon repair** — map out-of-vocabulary words onto near neighbours in the
#    training vocabulary, but only where the LM agrees and the edit distance is small.
#
# **Leakage control:** the language model and the lexicon are rebuilt *per fold* from
# the training portion only. Tuning against an LM that has memorised the validation
# sentences would produce weights that collapse on the leaderboard.

class CharLM:
    """Character n-gram model with stupid back-off - small, fast, no dependencies."""

    def __init__(self, texts: Sequence[str], order: int = 7, alpha: float = 0.4):
        self.order = order
        self.alpha = alpha
        self.ctx: dict[str, Counter] = defaultdict(Counter)
        self.ctx_total: Counter = Counter()
        self.uni: Counter = Counter()
        bos = "\x02"
        for t in texts:
            s = bos * (order - 1) + t + "\x03"
            for i in range(order - 1, len(s)):
                ch = s[i]
                self.uni[ch] += 1
                for k in range(order):
                    h = s[i - k:i]
                    self.ctx[h][ch] += 1
                    self.ctx_total[h] += 1
        self.n_uni = sum(self.uni.values())
        self.v_uni = len(self.uni) + 1

    def _logp(self, ctx: str, ch: str) -> float:
        penalty = 0.0
        for k in range(len(ctx), -1, -1):
            h = ctx[len(ctx) - k:] if k else ""
            c = self.ctx[h].get(ch, 0) if h in self.ctx else 0
            if c > 0:
                return math.log(c / self.ctx_total[h]) + penalty
            penalty += math.log(self.alpha)
        return math.log((self.uni.get(ch, 0) + 1) / (self.n_uni + self.v_uni)) + penalty

    def logprob(self, text: str) -> float:
        s = "\x02" * (self.order - 1) + text + "\x03"
        total = 0.0
        for i in range(self.order - 1, len(s)):
            total += self._logp(s[i - self.order + 1:i], s[i])
        return total

    def logprob_per_char(self, text: str) -> float:
        return self.logprob(text) / max(len(text) + 1, 1)


class Lexicon:
    def __init__(self, texts: Sequence[str]):
        self.freq = Counter(w for t in texts for w in t.split())
        self.vocab = set(self.freq)
        self.lower: dict[str, list[str]] = defaultdict(list)
        for w in self.vocab:
            self.lower[w.lower()].append(w)
        self.choices = sorted(self.vocab)   # sorted => deterministic tie-breaking
        self.mean_chars = float(np.mean([len(t) for t in texts])) if len(texts) else 60.0
        self.std_chars = float(np.std([len(t) for t in texts])) or 1.0

    def oov_rate(self, text: str) -> float:
        words = text.split()
        if not words:
            return 1.0
        return sum(w not in self.vocab for w in words) / len(words)


def match_case(source: str, target: str) -> str:
    """Give `target` the capitalisation pattern of `source`."""
    if source.isupper() and len(source) > 1:
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def repair_words(text: str, lex: Lexicon, min_sim: float = 88.0,
                 min_freq: int = 3, max_len_ratio: float = 0.34) -> str:
    """Snap out-of-vocabulary words onto close, frequent training-vocabulary words."""
    out = []
    for word in text.split():
        core = word.strip(".,;:()\"'")
        if not core or core in lex.vocab or len(core) < 3:
            out.append(word)
            continue
        exact_case = lex.lower.get(core.lower())
        if exact_case:  # only the capitalisation differs - keep what the model wrote
            out.append(word)
            continue
        budget = max(1, int(len(core) * max_len_ratio))
        # Look at several neighbours, not just the top one: near-ties are common
        # ("presence" vs "Presence") and only the frequent variant is a safe target.
        cands = rf_process.extract(
            core, lex.choices, scorer=Levenshtein.normalized_similarity,
            processor=str.lower, score_cutoff=min_sim / 100.0, limit=8,
        )
        viable = [
            (sim + 0.02 * math.log10(lex.freq[c] + 1), c)
            for c, sim, _ in cands
            if lex.freq[c] >= min_freq
            and Levenshtein.distance(core.lower(), c.lower()) <= budget
        ]
        if not viable:
            out.append(word)
            continue
        best = max(viable)[1]
        out.append(word.replace(core, match_case(core, best)))
    return " ".join(out)


def collapse_repeats(text: str, max_repeat: int = 3) -> str:
    """Truncate degenerate loops such as 'of the of the of the ...'."""
    words = text.split()
    for size in (1, 2, 3, 4):
        i = 0
        while i + size * (max_repeat + 1) <= len(words):
            block = words[i:i + size]
            reps = 1
            j = i + size
            while j + size <= len(words) and words[j:j + size] == block:
                reps += 1
                j += size
            if reps > max_repeat:
                words = words[:i + size * max_repeat] + words[j:]
            i += size
    return " ".join(words)

# ### 15.1 ROVER
#
# Hypotheses are aligned to the highest-weighted one with Levenshtein edit
# operations, giving a confusion network: one voting slot per pivot unit (a
# substitution or a deletion) plus one slot per gap (an insertion). Each slot is
# decided by weighted majority, so the fused output can contain a unit that only a
# minority of hypotheses proposed at that position while keeping the rest.

def _align_to_pivot(pivot: list, hyp: list) -> tuple[list, list]:
    """Return (aligned units per pivot slot, inserted runs per gap)."""
    align: list = [""] * len(pivot)
    inserts: list = [[] for _ in range(len(pivot) + 1)]
    i = j = 0
    for op in Levenshtein.editops(pivot, hyp):
        while i < op.src_pos:
            align[i] = hyp[j]
            i += 1
            j += 1
        if op.tag == "replace":
            align[i] = hyp[op.dest_pos]
            i += 1
            j += 1
        elif op.tag == "delete":
            align[i] = ""
            i += 1
        elif op.tag == "insert":
            inserts[i].append(hyp[op.dest_pos])
            j += 1
    while i < len(pivot):
        align[i] = hyp[j] if j < len(hyp) else ""
        i += 1
        j += 1
    return align, inserts


def rover_fuse(hyps: Sequence[str], weights: Sequence[float], level: str = "char") -> str:
    """Weighted confusion-network vote over aligned hypotheses."""
    pairs = [(h, w) for h, w in zip(hyps, weights) if isinstance(h, str)]
    if not pairs:
        return ""
    merged: dict[str, float] = defaultdict(float)
    for h, w in pairs:
        merged[h] += float(w)
    if len(merged) == 1:
        return next(iter(merged))

    items = sorted(merged.items(), key=lambda kv: -kv[1])
    total = sum(w for _, w in items)
    split = (lambda s: list(s)) if level == "char" else (lambda s: s.split())
    pivot = split(items[0][0])
    if not pivot:
        return items[0][0]

    slot_votes = [defaultdict(float) for _ in range(len(pivot))]
    gap_votes = [defaultdict(float) for _ in range(len(pivot) + 1)]
    for text, w in items:
        align, inserts = _align_to_pivot(pivot, split(text))
        for k, unit in enumerate(align):
            slot_votes[k][unit] += w
        for k, run in enumerate(inserts):
            gap_votes[k][tuple(run)] += w

    out: list = []
    for k in range(len(pivot) + 1):
        run, vote = max(gap_votes[k].items(), key=lambda kv: (kv[1], -len(kv[0])))
        if run and vote > total * 0.5:
            out.extend(run)
        if k < len(pivot):
            best_unit, best_vote = max(
                slot_votes[k].items(),
                key=lambda kv: (kv[1], kv[0] == pivot[k], -len(kv[0])),
            )
            _ = best_vote
            if best_unit:
                out.append(best_unit)
    return "".join(out) if level == "char" else " ".join(out)


def _test_rover() -> None:
    # Majority repairs a single-character error.
    assert rover_fuse(["hello world", "hello world", "hallo world"], [1, 1, 1]) == "hello world"
    # Majority repairs a missing character.
    assert rover_fuse(["presence", "presence", "presnce"], [1, 1, 1]) == "presence"
    # Majority removes a spurious insertion.
    assert rover_fuse(["the deed", "the deed", "the deedx"], [1, 1, 1]) == "the deed"
    # Weights dominate raw counts.
    assert rover_fuse(["abc", "abd", "abd"], [10.0, 1.0, 1.0]) == "abc"
    # Word level works too.
    assert rover_fuse(["a b c", "a b c", "a x c"], [1, 1, 1], level="word") == "a b c"
    # Degenerate inputs.
    assert rover_fuse([], []) == ""
    assert rover_fuse(["only"], [1.0]) == "only"
    # Fusion can beat every input (different errors in different places).
    fused = rover_fuse(["the quick brown fex", "the quick brawn fox", "the quick brown fox"], [1, 1, 1])
    assert fused == "the quick brown fox", fused
    print("[rover] all unit tests passed")


_test_rover()

# ==========================================================================
# p15_tune.py
# ==========================================================================

# ### 15.2 Feature extraction and weight search
#
# Candidate features, computed once and reused across every search iteration:
#
# | Feature | Meaning |
# |---|---|
# | `s_max`, `s_mean` | best / average beam sequence log-probability for this string |
# | `votes` | how many (beam, TTA view, fold) combinations produced this exact string — the single strongest signal in the table |
# | `lm` | character n-gram log-probability per character, fold-safe |
# | `oov` | fraction of words absent from the fold's training vocabulary |
# | `lenz` | absolute z-score of the length against the training length distribution |
# | `c_score`, `c_votes` | the same confidence / agreement signals from the kraken CTC member (Section 14.1) |
# | `c_sim` | similarity of the string to the CTC consensus reading — the anchor that penalises VLM output drifting away from the page |
#
# Scores from the two recognisers live on different scales, so they get separate
# columns; a string one family never produced gets that family's per-line floor.
# The whole search runs twice — VLM-only and VLM+CTC — and the CTC member is kept
# only if it wins out-of-fold.
#
# The search is a plain random search over the linear weights against the real
# competition metric. With ~5 free parameters and a few hundred evaluations that is
# both sufficient and much harder to over-fit than a gradient-based fit would be.

@dataclass
class PostConfig:
    w_score: float = 1.0
    w_votes: float = 0.35
    w_lm: float = 0.15
    w_oov: float = -0.25
    w_lenz: float = -0.02
    use_rover: bool = True
    rover_level: str = "char"
    rover_temp: float = 0.08
    repair: bool = False
    repair_sim: float = 90.0
    repair_freq: int = 3
    restrict_charset: bool = True
    collapse: bool = True
    # kraken CTC member (Section 14.1); all zero == the VLM-only behaviour.
    use_ctc: bool = False
    w_ctc_score: float = 0.0
    w_ctc_votes: float = 0.0
    w_ctc_sim: float = 0.0

    @property
    def families(self) -> tuple:
        return ("vlm", "ctc") if self.use_ctc else ("vlm",)


FOLD_LM: dict[int, CharLM] = {}
FOLD_LEX: dict[int, Lexicon] = {}


def fold_resources(fold: int) -> tuple[CharLM, Lexicon]:
    if fold not in FOLD_LM:
        texts = train_df.loc[train_df.fold != fold, "Target"].tolist() if fold >= 0 \
            else train_df["Target"].tolist()
        FOLD_LM[fold] = CharLM(texts, order=CFG.lm_order)
        FOLD_LEX[fold] = Lexicon(texts)
    return FOLD_LM[fold], FOLD_LEX[fold]


def clean_candidate(text: str, restrict: bool, collapse: bool) -> str:
    text = normalize_text(text)
    if collapse:
        text = collapse_repeats(text)
    if restrict:
        text = normalize_text(restrict_charset(text))
    return text


def family_of(cand: dict) -> str:
    return cand.get("family") or "vlm"


def build_features(nbest: dict, ids: Sequence[str], fold_of: dict,
                   restrict: bool = True, collapse: bool = True,
                   families: Sequence[str] = ("vlm", "ctc")) -> dict:
    """id -> {'texts': [...], 'M': (n_cand, 9) feature matrix}"""
    feats = {}
    for image_id in progress(ids, desc="features", unit="line"):
        lm, lex = fold_resources(fold_of.get(image_id, -1))
        agg: dict[str, dict] = {}
        for c in nbest.get(image_id, []):
            fam = family_of(c)
            if fam not in families:
                continue
            text = clean_candidate(c["text"], restrict, collapse)
            rec = agg.setdefault(text, {"vlm": [], "ctc": []})
            rec[fam].append(float(c["score"]))
        if not agg:
            agg[""] = {"vlm": [-50.0], "ctc": []}

        v_all = [x for r in agg.values() for x in r["vlm"]]
        c_all = [x for r in agg.values() for x in r["ctc"]]
        v_floor = (min(v_all) - 1.0) if v_all else 0.0
        c_floor = (min(c_all) - 1.0) if c_all else 0.0
        consensus = None
        if c_all:
            consensus = max((t for t, r in agg.items() if r["ctc"]),
                            key=lambda t: (len(agg[t]["ctc"]), max(agg[t]["ctc"])))

        texts, rows = [], []
        for text, rec in agg.items():
            v = np.asarray(rec["vlm"], dtype=np.float64)
            c = np.asarray(rec["ctc"], dtype=np.float64)
            rows.append([
                float(v.max()) if v.size else v_floor,
                float(v.mean()) if v.size else v_floor,
                math.log1p(v.size),
                lm.logprob_per_char(text),
                lex.oov_rate(text),
                abs((len(text) - lex.mean_chars) / lex.std_chars),
                float(c.max()) if c.size else c_floor,
                math.log1p(c.size),
                float(Levenshtein.normalized_similarity(text, consensus))
                if consensus is not None else 0.0,
            ])
            texts.append(text)
        feats[image_id] = {"texts": texts, "M": np.asarray(rows, dtype=np.float64)}
    return feats


def combined_scores(M: np.ndarray, pc: PostConfig) -> np.ndarray:
    return (pc.w_score * (0.5 * M[:, 0] + 0.5 * M[:, 1])
            + pc.w_votes * M[:, 2]
            + pc.w_lm * M[:, 3]
            + pc.w_oov * M[:, 4]
            + pc.w_lenz * M[:, 5]
            + pc.w_ctc_score * M[:, 6]
            + pc.w_ctc_votes * M[:, 7]
            + pc.w_ctc_sim * M[:, 8])


def apply_post(feats: dict, ids: Sequence[str], pc: PostConfig,
               lexicon_for: Callable[[str], Lexicon] | None = None) -> dict:
    out = {}
    for image_id in ids:
        f = feats[image_id]
        texts, M = f["texts"], f["M"]
        s = combined_scores(M, pc)
        if pc.use_rover and len(texts) > 1:
            z = s - s.max()
            w = np.exp(z / max(pc.rover_temp, 1e-3))
            w = w / w.sum()
            keep = w > 0.01
            if keep.sum() >= 2:
                pred = rover_fuse([t for t, k in zip(texts, keep) if k],
                                  w[keep].tolist(), level=pc.rover_level)
            else:
                pred = texts[int(np.argmax(s))]
        else:
            pred = texts[int(np.argmax(s))]
        if not pred.strip():
            order = np.argsort(-s)
            for j in order:
                if texts[j].strip():
                    pred = texts[j]
                    break
        if pc.repair and lexicon_for is not None:
            pred = repair_words(pred, lexicon_for(image_id),
                                min_sim=pc.repair_sim, min_freq=pc.repair_freq)
        out[image_id] = normalize_text(pred)
    return out


stage("post-processing weight search")
FOLD_OF = dict(zip(train_df.ID, train_df.fold))
HAS_CTC = bool(globals().get("HAS_CTC", False))


def oof_score(pc: PostConfig, feats: dict | None = None) -> float:
    preds = apply_post(feats if feats is not None else OOF_FEATS, OOF_IDS, pc,
                       lexicon_for=lambda i: fold_resources(FOLD_OF.get(i, -1))[1])
    return competition_score(OOF_REFS, [preds[i] for i in OOF_IDS])["score"]


def tune_post(use_ctc: bool, label: str) -> tuple:
    """Stages A-D for one member set. Returns (PostConfig, oof score, features)."""
    fams = ("vlm", "ctc") if use_ctc else ("vlm",)
    feats = build_features(OOF_NBEST, OOF_IDS, FOLD_OF, restrict=True, collapse=True,
                           families=fams)
    print(f"\n{'=' * 30} tuning: {label} {'=' * 30}")

    # ------------------------------------------------------------ stage A --
    print("[tune] stage A: rescoring weights (ROVER off)")
    base_pc = PostConfig(use_rover=False, repair=False, use_ctc=use_ctc)
    starts = [("default weights", base_pc)]
    if use_ctc:
        # Two hand-picked starting points so the search does not have to find them:
        # the VLM defaults anchored to the CTC reading, and CTC-only selection.
        starts += [
            ("defaults + CTC anchor", PostConfig(**{**asdict(base_pc), "w_ctc_sim": 1.0,
                                                    "w_ctc_votes": 0.3})),
            ("CTC only", PostConfig(w_score=0.0, w_votes=0.0, w_lm=0.1, w_oov=0.0,
                                    w_lenz=0.0, use_rover=False, repair=False,
                                    use_ctc=True, w_ctc_score=1.0, w_ctc_votes=1.0)),
        ]
    best_pc, best_val = None, -1.0
    for name, pc in starts:
        val = oof_score(pc, feats)
        print(f"  {name:<26} {val:.5f}")
        if val > best_val:
            best_pc, best_val = pc, val

    rng = np.random.default_rng(0)
    t0 = time.time()
    _bar = progress(range(CFG.lm_tune_iters), desc=f"tune A {label}", unit="trial")
    for it in _bar:
        trial = PostConfig(
            w_score=float(rng.uniform(0.0, 2.0)),
            w_votes=float(rng.uniform(0.0, 1.5)),
            w_lm=float(rng.uniform(0.0, 1.2)),
            w_oov=float(rng.uniform(-1.5, 0.1)),
            w_lenz=float(rng.uniform(-0.3, 0.05)),
            use_rover=False, repair=False, use_ctc=use_ctc,
        )
        if use_ctc:
            trial.w_ctc_score = float(rng.uniform(0.0, 1.5))
            trial.w_ctc_votes = float(rng.uniform(0.0, 1.5))
            trial.w_ctc_sim = float(rng.uniform(0.0, 4.0))
        val = oof_score(trial, feats)
        if val > best_val:
            best_pc, best_val = trial, val
            _bar.write(f"  it {it:4d}  {val:.5f}  "
                       f"score={trial.w_score:.2f} votes={trial.w_votes:.2f} "
                       f"lm={trial.w_lm:.2f} oov={trial.w_oov:.2f} lenz={trial.w_lenz:.3f}"
                       + (f" | ctc score={trial.w_ctc_score:.2f} votes={trial.w_ctc_votes:.2f}"
                          f" sim={trial.w_ctc_sim:.2f}" if use_ctc else ""))
        _bar.set_postfix_str(f"best {best_val:.5f}", refresh=False)
    _bar.close()
    print(f"[tune] stage A best {best_val:.5f} in {time.time()-t0:.0f}s")

    # ------------------------------------------------------------ stage B --
    print("[tune] stage B: ROVER")
    stage_b = [(best_pc, best_val, "off")]
    for level in ("char", "word"):
        for temp in (0.03, 0.08, 0.2, 0.5):
            trial = PostConfig(**{**asdict(best_pc), "use_rover": True,
                                  "rover_level": level, "rover_temp": temp})
            val = oof_score(trial, feats)
            stage_b.append((trial, val, f"{level} T={temp}"))
            print(f"  rover {level:<5} T={temp:<5} {val:.5f}")
    best_pc, best_val, chosen = max(stage_b, key=lambda kv: kv[1])
    print(f"[tune] stage B best: {chosen} -> {best_val:.5f}")

    # ------------------------------------------------------------ stage C --
    print("[tune] stage C: lexicon repair")
    stage_c = [(best_pc, best_val, "off")]
    for sim in (86.0, 90.0, 94.0):
        for freq in (2, 5, 15):
            trial = PostConfig(**{**asdict(best_pc), "repair": True,
                                  "repair_sim": sim, "repair_freq": freq})
            val = oof_score(trial, feats)
            stage_c.append((trial, val, f"sim={sim} freq={freq}"))
            print(f"  repair sim={sim} freq={freq:<3} {val:.5f}")
    best_pc, best_val, chosen = max(stage_c, key=lambda kv: kv[1])
    print(f"[tune] stage C best: {chosen} -> {best_val:.5f}")

    # ------------------------------------------------------------ stage D --
    print("[tune] stage D: charset restriction ablation")
    unrestricted = build_features(OOF_NBEST, OOF_IDS, FOLD_OF, restrict=False,
                                  collapse=True, families=fams)
    trial = PostConfig(**{**asdict(best_pc), "restrict_charset": False})
    val = oof_score(trial, unrestricted)
    print(f"  restrict_charset=True   {best_val:.5f}")
    print(f"  restrict_charset=False  {val:.5f}")
    if val > best_val:
        best_pc, best_val, feats = trial, val, unrestricted
        print("  keeping the unrestricted charset")
    return best_pc, best_val, feats


TUNED = {"vlm": tune_post(False, "VLM only")}
if HAS_CTC:
    TUNED["vlm+ctc"] = tune_post(True, "VLM + kraken CTC")

print("\n[tune] member-set gate (out-of-fold):")
for _k, (_pc, _v, _) in TUNED.items():
    print(f"  {_k:<10} {_v:.5f}")
_choice = max(TUNED, key=lambda k: TUNED[k][1])
POST, best_val, OOF_FEATS = TUNED[_choice]
if HAS_CTC:
    _delta = TUNED["vlm+ctc"][1] - TUNED["vlm"][1]
    verdict = "KEPT" if _choice == "vlm+ctc" else "DROPPED"
    print(f"[tune] kraken CTC member {verdict} ({_delta:+.5f} OOF)")
    CTC_REPORT["gate_delta"] = _delta
    CTC_REPORT["gate_kept"] = _choice == "vlm+ctc"
    # All three weights are fitted on the same OOF rows the gate reads, so a
    # delta within ~1e-4 is not evidence either way.
    if abs(_delta) < 1e-4:
        print("[tune] note: the difference is within tuning noise")

with open(OUT / "post_config.json", "w") as fh:
    json.dump(asdict(POST), fh, indent=2)
print(f"\n[tune] final post-processing config:\n{json.dumps(asdict(POST), indent=2)}")

# Incremental contribution of each stage, measured out-of-fold.
_vlm_feats = TUNED["vlm"][2]
_vlm_post = TUNED["vlm"][0]
ladder = [
    ("raw best beam", _vlm_feats, PostConfig(w_score=1.0, w_votes=0.0, w_lm=0.0,
                                             w_oov=0.0, w_lenz=0.0, use_rover=False,
                                             repair=False, restrict_charset=False,
                                             collapse=False)),
    ("+ cleanup (charset, loops)", _vlm_feats, PostConfig(w_score=1.0, w_votes=0.0,
                                                          w_lm=0.0, w_oov=0.0,
                                                          w_lenz=0.0, use_rover=False,
                                                          repair=False)),
    ("+ vote/LM rescoring", _vlm_feats,
     PostConfig(**{**asdict(_vlm_post), "use_rover": False, "repair": False})),
    ("+ ROVER", _vlm_feats, PostConfig(**{**asdict(_vlm_post), "repair": False})),
    ("+ lexicon repair", _vlm_feats, _vlm_post),
]
if HAS_CTC:
    ladder.append(("+ kraken CTC member" + ("" if POST.use_ctc else " (rejected)"),
                   TUNED["vlm+ctc"][2], TUNED["vlm+ctc"][0]))
rows = []
prev = None
for name, feats_, pc in ladder:
    val = oof_score(pc, feats_)
    rows.append({"stage": name, "oof_score": round(val, 5),
                 "delta": None if prev is None else round(val - prev, 5)})
    prev = val
LADDER = pd.DataFrame(rows)
print(LADDER.to_string(index=False))
LADDER.to_csv(OUT / "postprocessing_ladder.csv", index=False)

# Error inspection: the worst out-of-fold lines, to decide where to spend next.
_final_oof = apply_post(OOF_FEATS, OOF_IDS, POST,
                        lexicon_for=lambda i: fold_resources(FOLD_OF.get(i, -1))[1])
_err = error_rates(OOF_REFS, [_final_oof[i] for i in OOF_IDS], "char")["per_sample"]
_worst_idx = np.argsort(-_err)[:12]
print("worst out-of-fold lines\n" + "-" * 78)
for j in _worst_idx:
    print(f"CER {_err[j]:.2f}  id={OOF_IDS[j]}")
    print(f"  ref  {OOF_REFS[j]}")
    print(f"  pred {_final_oof[OOF_IDS[j]]}")
print("-" * 78)
print(f"lines with CER = 0     : {(_err == 0).mean():.1%}")
print(f"lines with CER > 0.5   : {(_err > 0.5).mean():.1%}")

# ==========================================================================
# p16_submit.py
# ==========================================================================

# ## 16. Optional: self-training on the test images
#
# Pseudo-labelling the test set and training on it is a large, reliable gain in most
# OCR competitions. **It is disabled here on purpose.** The organiser thread on
# self-training was still open when this notebook was written, and the neighbouring
# ruling on StackMix was explicit that test images may not be used. Turning this on
# without a written ruling risks disqualification.
#
# If and only if the organisers confirm it is allowed, set
# `CFG.use_pseudo_labels = True`: high-confidence test predictions (mean beam score
# above a percentile threshold, with unanimous agreement across TTA views) are added
# to the training set with a lower loss weight, and the folds are retrained.

def build_pseudo_labels(nbest: dict, min_agreement: float = 1.0,
                        score_percentile: float = 60.0) -> pd.DataFrame:
    rows = []
    for image_id, cands in nbest.items():
        texts = [normalize_text(c["text"]) for c in cands]
        scores = [c["score"] for c in cands]
        if not texts:
            continue
        top = Counter(texts).most_common(1)[0]
        agreement = top[1] / len(texts)
        rows.append({"ID": image_id, "Target": top[0],
                     "agreement": agreement, "score": float(np.mean(scores))})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    cut = np.percentile(df["score"], score_percentile)
    keep = df[(df.agreement >= min_agreement) & (df.score >= cut) &
              (df.Target.str.len() > 15)].copy()
    print(f"[pseudo] {len(keep)}/{len(df)} test lines pass the confidence gate")
    return keep


if CFG.use_pseudo_labels:
    PSEUDO = build_pseudo_labels({i: [c for c in v if family_of(c) == "vlm"]
                                  for i, v in TEST_NBEST.items()})
    print("!! Pseudo-labelling is enabled. Confirm the organisers permit it before "
          "submitting a solution trained this way.")
    # Retrain fold 0 with the pseudo-labelled lines appended, then re-run Section 14.
    _pseudo_train = pd.concat([
        train_df[train_df.fold != CFG.train_folds[0]][["ID", "Target"]],
        PSEUDO[["ID", "Target"]],
    ], ignore_index=True)
    RUNS.append(train_fold(CFG.train_folds[0], tag="pseudo_fold0",
                           train_override=_pseudo_train))
else:
    print("[pseudo] disabled - see the note above")

# ## 17. Submission
#
# The tuned post-processing is applied to the pooled test N-best lists using a
# language model and lexicon built from the **entire** training set (no fold
# restriction is needed here — the test labels were never part of it).

stage("submission")
TEST_IDS = test_df.ID.tolist()
TEST_FOLD_OF = {i: -1 for i in TEST_IDS}   # -1 => resources built from all of Train.csv
fold_resources(-1)

TEST_FEATS = build_features(TEST_NBEST, TEST_IDS, TEST_FOLD_OF,
                            restrict=POST.restrict_charset, collapse=POST.collapse,
                            families=POST.families)
print(f"[submit] members: {' + '.join(POST.families)}"
      f"{'' if POST.use_ctc or not HAS_CTC else '  (kraken CTC rejected by the OOF gate)'}")
TEST_PRED = apply_post(TEST_FEATS, TEST_IDS, POST,
                       lexicon_for=lambda i: FOLD_LEX[-1])

submission = pd.DataFrame({"ID": TEST_IDS, "Target": [TEST_PRED[i] for i in TEST_IDS]})

# ------------------------------------------------------------------ checks --
problems = []
if len(submission) != len(sample_sub):
    problems.append(f"row count {len(submission)} != sample {len(sample_sub)}")
if set(submission.ID) != set(sample_sub.ID):
    missing = set(sample_sub.ID) - set(submission.ID)
    extra = set(submission.ID) - set(sample_sub.ID)
    problems.append(f"ID mismatch (missing {len(missing)}, extra {len(extra)})")
if submission.Target.isna().any():
    problems.append("NaN predictions")
n_empty = int((submission.Target.str.strip() == "").sum())
if n_empty:
    problems.append(f"{n_empty} empty predictions (scored as fully incorrect)")
    # VLM beams only: CTC confidences live on a different scale.
    fallback = top1({i: [c for c in cands if family_of(c) == "vlm"] or cands
                     for i, cands in TEST_NBEST.items()})
    mask = submission.Target.str.strip() == ""
    submission.loc[mask, "Target"] = [normalize_text(fallback.get(i, "the")) or "the"
                                      for i in submission.loc[mask, "ID"]]
    submission.loc[submission.Target.str.strip() == "", "Target"] = "the"

bad_chars = Counter(ch for t in submission.Target for ch in t if ch not in ALLOWED)
if bad_chars:
    problems.append(f"characters outside the training charset: {dict(bad_chars)}")

print("submission checks:", "OK" if not problems else "")
for p in problems:
    print("  !!", p)

lengths = submission.Target.str.len()
print(f"\nlength: median {lengths.median():.0f} (train {train_df.Target.str.len().median():.0f}) "
      f"| p10 {lengths.quantile(0.1):.0f} | p90 {lengths.quantile(0.9):.0f}")
words = submission.Target.str.split().apply(len)
print(f"words:  median {words.median():.0f} (train {train_df.n_words.median():.0f})")
print(f"OOV rate vs training vocabulary: {np.mean([FOLD_LEX[-1].oov_rate(t) for t in submission.Target]):.1%} "
      f"(train self-OOV would be 0%)")

SUB_PATH = WORK / "submission.csv"
submission.to_csv(SUB_PATH, index=False)
print(f"\nwrote {SUB_PATH} ({len(submission)} rows)")
print(submission.head(10).to_string(index=False))

# Keep a scored, reproducible record of the run alongside the submission, and mirror
# it to the hub so the score and the weights that produced it never separate.
manifest = {
    "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    "run_id": RUN_FINGERPRINT["run_id"],
    "fingerprint": RUN_FINGERPRINT,
    "gpu": GPU_NAME,
    "model": CFG.model_id,
    "model_licence": CFG_LICENCE,
    "finetune_mode": CFG.finetune_mode,
    "folds_trained": [r["tag"] for r in RUNS],
    "per_fold_val_score": {r["tag"]: r["best"]["score"] for r in RUNS},
    "grpo_scores": {r["tag"]: r.get("grpo_scores") for r in RUNS if r.get("grpo_scores")},
    "grpo_accepted": [r["tag"] for r in RUNS if r.get("grpo_path")],
    "oof_baseline_score": round(BASELINE["score"], 5),
    "oof_final_score": round(best_val, 5),
    "post_config": asdict(POST),
    "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(CFG).items()},
    "stackmix_enabled": bool(CFG.stackmix),
    "pseudo_labels_used": bool(CFG.use_pseudo_labels),
    "grpo_used": bool(CFG.run_grpo),
    "ctc_branch": CTC_STATUS,
    "ctc_model": CFG.ctc_model_id if CFG.ctc_branch else None,
    "ctc_model_licence": CTC_LICENCE,
    "ctc_used_in_submission": bool(POST.use_ctc),
    "ctc_report": CTC_REPORT,
    "tuned_member_sets": {k: round(v[1], 5) for k, v in TUNED.items()},
    "hub_repo": CFG.hf_repo_id or None,
    "hub_errors": HUB.errors,
}
with open(OUT / "run_manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2, default=str)
with open(OUT / "post_config.json", "w") as fh:
    json.dump(asdict(POST), fh, indent=2)

SUMMARY_TB = TBLogger("summary", enabled=CFG.tensorboard)
SUMMARY_TB.scalars({"oof_raw": BASELINE["score"], "oof_fused": best_val,
                    "oof_cer": BASELINE["cer"], "oof_wer": BASELINE["wer"]},
                   0, prefix="final/")
for _i, _r in enumerate(RUNS):
    SUMMARY_TB.scalar("final/fold_val_score", _r["best"]["score"], _i)
SUMMARY_TB.close()

HUB.push_file(OUT / "run_manifest.json", "run_manifest.json", f"manifest {best_val:.5f}")
HUB.push_file(OUT / "post_config.json", "post_config.json", "tuned post-processing")
if CFG.hf_push_submission:
    HUB.push_file(SUB_PATH, "submission.csv", f"submission oof {best_val:.5f}")
push_tensorboard("summary", "final metrics")
HUB.write_card(model_card(
    CFG, CFG_LICENCE, RUN_FINGERPRINT, RUNS,
    scores={
        "OOF score (after fusion)": f"{best_val:.5f}",
        "OOF score (raw top-1)": f"{BASELINE['score']:.5f}",
        "weighted CER / WER": f"{BASELINE['cer']:.4f} / {BASELINE['wer']:.4f}",
        "kraken PP-OCRv6 member": ("used" if POST.use_ctc else "not used")
                                  + f" ({CTC_STATUS})",
    },
))

print("=" * 78)
print("RUN SUMMARY")
print("=" * 78)
print(f"run id              {RUN_FINGERPRINT['run_id']}  (config {RUN_FINGERPRINT['config_hash']})")
print(f"backbone            {CFG.model_id}  ({CFG_LICENCE})")
print(f"fine-tune mode      {CFG.finetune_mode}"
      f"{'  + GRPO' if any(r.get('grpo_path') for r in RUNS) else ''}")
print(f"folds trained       {len(RUNS)}  ({', '.join(r['tag'] for r in RUNS)})")
for r in RUNS:
    extra = ""
    if r.get("grpo_scores"):
        kept = "kept" if r.get("grpo_path") else "rejected"
        extra = f"  [grpo {r['grpo_scores']['delta']:+.5f} {kept}]"
    print(f"  {r['tag']:<24} val {r['best']['score']:.5f} "
          f"({r['best']['source']}, step {r['best']['step']}, {r.get('minutes', 0):.0f} min){extra}")
print(f"kraken CTC branch   {CTC_STATUS}")
if CTC_REPORT.get("zero_shot"):
    print(f"  zero-shot PP-OCRv6          {CTC_REPORT['zero_shot']['score']:.5f}")
if CTC_REPORT.get("oof_top1"):
    print(f"  fine-tuned CTC top-1 OOF    {CTC_REPORT['oof_top1']['score']:.5f}")
if "gate_delta" in CTC_REPORT:
    print(f"  fusion gate                 {'kept' if CTC_REPORT['gate_kept'] else 'dropped'}"
          f" ({CTC_REPORT['gate_delta']:+.5f} OOF)")
print(f"OOF before post-processing   {BASELINE['score']:.5f}")
print(f"OOF after post-processing    {best_val:.5f}   ({best_val - BASELINE['score']:+.5f})")
print(f"submission                   {SUB_PATH}")
if HUB.enabled:
    print(f"hub                          https://huggingface.co/{CFG.hf_repo_id}")
    print(f"metrics                      https://huggingface.co/{CFG.hf_repo_id}/tensorboard")
    print("replay this run              CFG.hf_resume = True  (skips training)")
print("=" * 78)

STAGE_TIMES = stage_summary()
STAGE_TIMES.to_csv(OUT / "stage_times.csv", index=False)

NOTIFY.finished(best_val, BASELINE["score"], SUB_PATH)

# ## 18. Where the remaining points are
#
# Ordered by expected gain per GPU-hour, based on the diagnostics printed above.
#
# 1. **Train more folds.** Ensembling is the most reliable lever in this pipeline —
#    each additional fold adds independent hypotheses to the ROVER vote. Go from
#    `train_folds = (0,)` to all five before tuning anything else.
# 2. **Check the oracle gap** printed at the end of `Section 14`. If the oracle over
#    the N-best list is far above the fused score, the ranking is the bottleneck:
#    raise `n_best`, add TTA views, and re-run the weight search. If the oracle is
#    close to the fused score, the *model* is the bottleneck: train longer, raise
#    `line_height`, or move to a larger backbone.
# 3. **Resolution.** `line_height` is the single most influential image knob. If VRAM
#    allows, 192 with `max_line_width = 3840` gives the vision tower noticeably more
#    signal per character, at roughly 1.4x the step time.
# 4. **Seed ensembling.** Two seeds on the same fold produce genuinely different error
#    patterns and fuse almost as well as two different folds, without shrinking the
#    training set. Run `train_fold(0, tag="fold0_seedB")` with `CFG.seed` changed and
#    append it to `RUNS`; the ROVER vote picks the diversity up for free.
# 5. **Mode diversity.** `CFG.finetune_mode = "full"` on a 4B backbone produces a
#    model that fails differently from the 8B LoRA one. Two *different* models fuse
#    better than two runs of the same recipe, so this is worth more as an ensemble
#    member than as a replacement.
# 6. **GRPO steps.** GRPO is on by default and self-gating, so the question is only
#    how much of it to buy: `grpo_steps` is the knob, and the printed
#    `[grpo] SFT x vs GRPO y` line tells you whether more would pay. If the delta is
#    positive and still growing at 300 steps, raise it.
# 7. **Read the kraken CTC verdict.** The summary prints the zero-shot and
#    fine-tuned PP-OCRv6 scores and whether the fusion gate kept it. If the
#    fine-tuned CTC top-1 is within a few thousandths of the VLM, it is a genuine
#    second opinion: try `ctc_epochs=80`, `ctc_line_height=128` with
#    `ctc_max_width=5120`, or StackMix lines in its training set. If it is far
#    behind, it still earns its place through the `c_sim` anchor on the lines where
#    the VLM loops — check the gate delta rather than dropping it on instinct.
# 8. **Longest lines.** Inspect the worst-CER examples printed above: if they cluster
#    at high aspect ratios, they are being squashed by `max_line_width` and deserve a
#    dedicated higher-budget decoding pass.


# ==========================================================================
print("=" * 78)
print(f"[script] complete. log: {LOG_PATH}")
if "NOTIFY" in dir() and getattr(NOTIFY, "enabled", False):
    print(f"[script] {NOTIFY.sent} notification(s) sent to {NOTIFY.to}")
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.flush()
    except Exception:
        pass
