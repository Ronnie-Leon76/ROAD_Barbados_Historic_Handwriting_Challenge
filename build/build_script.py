"""Assemble a headless training script from the same percent-format source parts.

`build_notebook.py` emits the interactive deliverable; this emits `run_pipeline.py`,
a single self-contained script for unattended runs on a rented pod. Both read the
same `build/parts/p*.py`, so the script and the notebook cannot drift apart.

The generated script adds three things a notebook does not need:

* argument parsing, so folds/mode/epochs can be set without editing source;
* a non-interactive matplotlib backend and unbuffered, timestamped logging to a file;
* a crash handler that emails the traceback before the process exits.

Usage:
    python build/build_script.py                    # -> run_pipeline.py (VLM)
    python build/build_script.py --profile kraken   # -> run_kraken.py
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from build_notebook import parse_percent_file, read_manifest

DOC_VLM = '''"""R.O.A.D. Barbados Historic Handwriting Challenge - unattended training run.

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
"""'''

DOC_KRAKEN = '''"""R.O.A.D. Barbados Historic Handwriting Challenge - kraken PP-OCRv6 run.

GENERATED FILE - do not edit. Regenerate with:

    python build/build_script.py --profile kraken

Source of truth: build/kraken_manifest.txt (the same cells as
ROAD_Barbados_HTR_Kraken.ipynb).

Examples:
    python run_kraken.py                                    # 5 folds x 2 seeds
    python run_kraken.py --folds 0 --seeds 42 --epochs 10   # quick pass
    python run_kraken.py --hf-repo me/road-htr --vlm-preds artifacts/preds
    python run_kraken.py --set lm_beam=24 --set kraken_epochs=60

Background:
    nohup python -u run_kraken.py > logs/kraken.out 2>&1 &
"""'''

ARGS_VLM = '''
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
'''

OVERRIDES_VLM = '''
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
'''

ARGS_KRAKEN = '''
def _args():
    ap = argparse.ArgumentParser(
        description="Fine-tune kraken PP-OCRv6 per fold, decode, fuse, submit.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--folds", type=int, nargs="+", metavar="N",
                    help="folds to train (default: all)")
    ap.add_argument("--seeds", type=int, nargs="+", metavar="S",
                    help="one model per seed per fold; posteriors are averaged")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--hf-repo", metavar="ID", help="hub repo for weights + predictions")
    ap.add_argument("--vlm-preds", metavar="DIR",
                    help="directory with the VLM notebook's oof_*.csv / test_*.csv")
    ap.add_argument("--email", metavar="ADDR", help="override NOTIFY_EMAIL_TO")
    ap.add_argument("--no-stackmix", dest="stackmix", action="store_false", default=None)
    ap.add_argument("--no-soup", dest="soup", action="store_false", default=None,
                    help="skip checkpoint averaging")
    ap.add_argument("--no-tensorboard", dest="tensorboard", action="store_false",
                    default=None)
    ap.add_argument("--no-email", dest="notify", action="store_false", default=None)
    ap.add_argument("--resume-from-hub", action="store_true",
                    help="download weights instead of training")
    ap.add_argument("--fast", action="store_true",
                    help="1 fold, 1 seed, 2 epochs, one view - a plumbing check")
    ap.add_argument("--install", action="store_true",
                    help="pip install the light dependencies first (new pod)")
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--progress-interval", type=float, default=30.0, metavar="SEC",
                    help="seconds between progress-bar refreshes in the log")
    ap.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE",
                    help="set any CFG field, repeatable")
    return ap.parse_args()
'''

OVERRIDES_KRAKEN = '''
CLI_OVERRIDES: dict = {}
if ARGS.folds is not None:
    CLI_OVERRIDES["train_folds"] = tuple(ARGS.folds)
if ARGS.seeds is not None:
    CLI_OVERRIDES["kraken_seeds"] = tuple(ARGS.seeds)
for flag, field in (("epochs", "epochs"), ("lr", "lr"), ("hf_repo", "hf_repo_id"),
                    ("vlm_preds", "vlm_preds_dir"), ("stackmix", "stackmix"),
                    ("soup", "kraken_soup"), ("tensorboard", "tensorboard"),
                    ("notify", "notify_email")):
    value = getattr(ARGS, flag, None)
    if value is not None:
        CLI_OVERRIDES[field] = value
if ARGS.resume_from_hub:
    CLI_OVERRIDES["hf_resume"] = True
if ARGS.fast:
    CLI_OVERRIDES.update({"train_folds": (0,), "kraken_seeds": (42,), "epochs": 2,
                          "kraken_tta_modes": ("identity",), "lm_beam": 8,
                          "lm_alpha_grid": (0.0, 0.5), "lm_beta_grid": (0.0, 1.0),
                          "kraken_soup": False, "lm_tune_iters": 40})
'''

PROFILES = {
    "vlm": {"doc": DOC_VLM, "args": ARGS_VLM, "overrides": OVERRIDES_VLM,
            "out": "run_pipeline.py", "manifest": None},
    "kraken": {"doc": DOC_KRAKEN, "args": ARGS_KRAKEN, "overrides": OVERRIDES_KRAKEN,
               "out": "run_kraken.py", "manifest": "kraken_manifest.txt"},
}

HEADER = '''#!/usr/bin/env python
@@DOC@@

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

@@ARGS@@

class _Tee:
    """Mirror stdout/stderr to a log file so a detached run stays inspectable.

    Progress bars redraw with a carriage return, which a terminal renders in place
    but a file would accumulate into one enormous line. Treating \\r as a line break
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
        for i, chunk in enumerate(data.replace("\\r\\n", "\\n").replace("\\r", "\\n").split("\\n")):
            if i:
                self.file.write("\\n")
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
                    f"The run failed before training began.\\n\\n{detail}"[:8000])
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
@@OVERRIDES@@
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
'''

FOOTER = '''

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
'''

# The notifier is only defined part-way through the run, so the crash handler is
# installed as soon as it exists rather than at the top of the file.
CRASH_HOOK = '''
# --- installed by build_script.py: email any uncaught exception ---------------
NOTIFY.install_crash_handler()
'''


def render_header(profile: str) -> str:
    spec = PROFILES[profile]
    return (HEADER.replace("@@DOC@@", spec["doc"])
                  .replace("@@ARGS@@", spec["args"])
                  .replace("@@OVERRIDES@@", spec["overrides"].strip("\n")))


def build(parts_dir: pathlib.Path, out_path: pathlib.Path, profile: str = "vlm",
          manifest: pathlib.Path | None = None) -> int:
    part_files = read_manifest(manifest) if manifest else sorted(parts_dir.glob("p*.py"))
    if not part_files:
        raise SystemExit(f"No parts found in {parts_dir}")

    chunks: list[str] = [render_header(profile)]
    n_cells = 0
    for path in part_files:
        text = path.read_text(encoding="utf-8")
        try:
            compile(text, str(path), "exec")
        except SyntaxError as exc:
            raise SystemExit(f"SyntaxError in {path.name}:{exc.lineno}: {exc.msg}") from exc
        chunks.append(f"\n# {'=' * 74}\n# {path.name}\n# {'=' * 74}\n")
        for cell_type, src in parse_percent_file(text):
            if cell_type == "markdown":
                body = "\n".join(f"# {ln}".rstrip() for ln in src.split("\n"))
                chunks.append(f"\n{body}\n")
            else:
                chunks.append(f"\n{src}\n")
                n_cells += 1
        if path.name.startswith("p02c"):
            chunks.append(CRASH_HOOK)

    chunks.append(FOOTER)
    source = "".join(chunks)
    try:
        compile(source, str(out_path), "exec")
    except SyntaxError as exc:
        raise SystemExit(f"generated script is invalid at line {exc.lineno}: {exc.msg}")
    out_path.write_text(source, encoding="utf-8")
    return n_cells


def main() -> int:
    root = pathlib.Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", default=str(root / "parts"))
    ap.add_argument("--profile", choices=sorted(PROFILES), default="vlm",
                    help="which command line / part list to generate")
    ap.add_argument("--out", default=None,
                    help="output script (default: run_pipeline.py / run_kraken.py)")
    ap.add_argument("--manifest", default=None,
                    help="ordered part list (default: the profile's manifest)")
    args = ap.parse_args()

    spec = PROFILES[args.profile]
    out = pathlib.Path(args.out or (root.parent / spec["out"]))
    manifest = args.manifest or (root / spec["manifest"] if spec["manifest"] else None)
    n_cells = build(pathlib.Path(args.parts), out, args.profile,
                    pathlib.Path(manifest) if manifest else None)
    n_lines = len(out.read_text(encoding="utf-8").split("\n"))
    print(f"wrote {out}: {n_cells} cells, {n_lines} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
