# %% [markdown]
# ## 2. Config
#
# The levers that actually change a score, in the order to touch them:
#
# 1. `train_folds = (0, 1, 2, 3, 4)` — already the default. Five CTC models are
#    cheap and the ROVER vote is the most reliable point on this dataset.
# 2. `also_train_small = True` — a second architecture in the same family.
# 3. `resize = "new"` — collapse the 44-language softmax onto Train.csv. Switch to
#    `"union"` only if zero-shot already reads the challenge charset well.
# 4. `vlm_test_csv` — path to the Qwen `submission.csv`. Blended at test time
#    when present; OOF-tuned if `vlm_oof_csv` is also present.
#
# Batch size and `--max-width` are **not** levers: `tune_for_gpu()` sets them
# from VRAM so an L40S is fed and a 16 GB card does not OOM. Ketos is always
# `-q fixed` with no `--freeze-backbone` (ppocrv6).

# %%
def load_dotenv(path: str | Path = ".env") -> list:
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
    work_dir: str = "."
    data_dir: str = "data"
    image_subdir: str = "images"
    out_dir: str = "artifacts_kraken"
    train_csv: str = "Train.csv"
    test_csv: str = "Test.csv"
    sample_sub_csv: str = "SampleSubmission.csv"
    images_url: str = "https://storage.googleapis.com/road-handwriting/images.zip"

    model_id: str = "small-models-for-glam/kraken-ppocrv6-medium"
    model_file: str = "medium.safetensors"
    small_model_id: str = "small-models-for-glam/kraken-ppocrv6-small"
    small_model_file: str = "small.safetensors"
    also_train_small: bool = True
    kraken_version: str = "7.1.1"

    n_folds: int = 5
    train_folds: tuple = (0, 1, 2, 3, 4)
    fold_seed: int = 20260703
    seed: int = 42
    deterministic: bool = True

    # ketos. PP-OCRv6 rejects `--freeze-backbone` (vgsl-only) and crashes on
    # `-q early`: Lightning then sets max_epochs=-1, `estimated_stepping_batches`
    # is inf, and the default AdamW+Muon cosine schedule does int(inf)
    # (kraken#792). Use a fixed epoch budget. Leave optimizer empty so ketos
    # keeps the architecture default (AdamW+Muon).
    resize: str = "new"                 # new | union  (see markdown above)
    epochs: int = 50
    min_epochs: int = 15
    quit: str = "fixed"                 # fixed | early  (early is unsafe on ppocrv6)
    lr: float = 3e-4
    warmup: int = 200
    freeze_backbone: int = 0            # 0 = omit; ppocrv6 does not support it
    batch_size: int = 8                 # overwritten by tune_for_gpu()
    schedule: str = "cosine"
    optimizer: str = ""                 # "" = architecture default (AdamW+Muon)
    augment: bool = True
    line_height: int = 128              # PP-OCRv6 medium native height
    max_line_width: int = 2560          # overwritten by tune_for_gpu(); static pad
    min_line_width: int = 64
    min_line_height: int = 48
    max_upscale: float = 3.0
    patch_grid: int = 1
    ink_crop_horizontal: bool = True
    ink_crop_vertical: bool = False
    deskew: bool = True
    deskew_max_deg: float = 3.0
    clahe: bool = True
    background_flatten: bool = True

    tta_modes: tuple = ("identity", "dark", "light", "sharp")
    infer_batch: int = 16
    dataloader_workers: int = 2

    lm_order: int = 7
    lm_tune_iters: int = 400

    # Path to the Qwen submission (optional ensemble). Leave empty to skip.
    vlm_test_csv: str = "submission_road_htr.csv"
    vlm_oof_csv: str = ""               # ID,Target per OOF line, if you have it
    vlm_blend: float = 0.35             # used only when OOF VLM is missing

    hf_repo_id: str = ""
    hf_token_env: str = "HF_TOKEN"
    hf_private: bool = True
    notify_email: bool = True            # same .env SMTP keys as the Qwen notebook
    tensorboard: bool = True


CFG = Config()
if not CFG.hf_repo_id:
    CFG.hf_repo_id = os.environ.get("HF_REPO_ID", "").strip()
WORK = Path(CFG.work_dir).resolve()
OUT = WORK / CFG.out_dir
OUT.mkdir(parents=True, exist_ok=True)

PERMISSIVE = {
    "apache-2.0", "mit", "bsd", "bsd-2-clause", "bsd-3-clause",
    "cc0-1.0", "cc-by-4.0", "unlicense", "isc",
}
KNOWN_LICENCES = {
    "small-models-for-glam/kraken-ppocrv6-medium": ("apache-2.0", True),
    "small-models-for-glam/kraken-ppocrv6-small": ("apache-2.0", True),
    "small-models-for-glam/kraken-ppocrv6-tiny": ("apache-2.0", True),
}


def check_licence(model_id: str) -> str:
    licence, verdict = KNOWN_LICENCES.get(model_id, (None, None))
    source = "static table"
    try:
        from huggingface_hub import model_info
        card = (model_info(model_id).cardData or {})
        hub = str(card.get("license") or card.get("license_name") or "").strip().lower()
        if hub:
            licence, source, verdict = hub, "huggingface hub", hub in PERMISSIVE
    except Exception as exc:
        print(f"[licence] hub lookup failed for {model_id} ({type(exc).__name__})")
    if not licence:
        raise RuntimeError(f"unknown licence for {model_id}")
    print(f"[licence] {model_id}: {licence} -> {'OK' if verdict else 'BLOCKED'} ({source})")
    if not verdict:
        raise RuntimeError(f"{model_id} is '{licence}', which the organisers disallow")
    return licence


CFG_LICENCE = check_licence(CFG.model_id)
SMALL_LICENCE = check_licence(CFG.small_model_id) if CFG.also_train_small else None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)


def tune_for_gpu() -> None:
    """Size the PP-OCRv6 static-pad rectangle to the card. Mutates CFG in place.

    The NRTR auxiliary head is the VRAM hog, not the 16 M CTC backbone. Batch 48
    at width 3200 OOMs on an L40S; 24 × 2560 trains. Do not start from the
    highest batch that *might* fit — the first OOM retry costs ~20 s and a
    compiled CUDA context.
    """
    vram = float(GPU_VRAM_GB)
    if vram >= 40:
        CFG.batch_size, CFG.max_line_width = 24, 2560
        CFG.dataloader_workers, CFG.infer_batch = 4, 32
    elif vram >= 22:
        CFG.batch_size, CFG.max_line_width = 16, 2560
        CFG.dataloader_workers, CFG.infer_batch = 4, 24
    elif vram >= 14:
        CFG.batch_size, CFG.max_line_width = 8, 2560
        CFG.dataloader_workers, CFG.infer_batch = 2, 16
    else:
        CFG.batch_size, CFG.max_line_width = 4, 1920
        CFG.dataloader_workers, CFG.infer_batch = 2, 8
    CFG.quit = "fixed"
    CFG.freeze_backbone = 0
    print(f"[hw] {vram:.0f} GB -> batch {CFG.batch_size}  max_width {CFG.max_line_width}  "
          f"workers {CFG.dataloader_workers}")


tune_for_gpu()
print(f"[cfg] {CFG.model_id} | folds {CFG.train_folds} | resize {CFG.resize} "
      f"| {CFG.epochs} epochs {CFG.quit} | lr {CFG.lr:g} | batch {CFG.batch_size}")

# %%
from tqdm.auto import tqdm

INTERACTIVE = bool(getattr(sys.stdout, "isatty", lambda: False)())
PROGRESS_INTERVAL = float(os.environ.get("PROGRESS_INTERVAL", "30"))
RUN_T0 = time.time()


def fmt_hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def progress(iterable=None, *, total=None, desc="", unit="it", leave=True, **kw):
    opts = dict(desc=desc, unit=unit, leave=leave, total=total,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]{postfix}")
    if INTERACTIVE:
        opts.update(ncols=100, mininterval=0.2)
    else:
        opts.update(ncols=80, ascii=True, mininterval=PROGRESS_INTERVAL, miniters=1)
    opts.update(kw)
    return tqdm(iterable, **opts)


STAGES: list[dict] = []


def stage(name: str) -> None:
    now = time.time()
    if STAGES:
        STAGES[-1]["seconds"] = now - STAGES[-1]["t0"]
    STAGES.append({"name": name, "t0": now, "seconds": None})
    print(f"\n{'=' * 78}\n[stage {len(STAGES)}] {name}    (+{fmt_hms(now - RUN_T0)})\n{'=' * 78}")


def stage_summary() -> pd.DataFrame:
    now = time.time()
    if STAGES and STAGES[-1]["seconds"] is None:
        STAGES[-1]["seconds"] = now - STAGES[-1]["t0"]
    total = max(now - RUN_T0, 1e-6)
    df = pd.DataFrame([{"stage": s["name"], "minutes": round((s["seconds"] or 0) / 60, 1),
                        "share": f"{100 * (s['seconds'] or 0) / total:5.1f}%"} for s in STAGES])
    print(f"\nwall clock {fmt_hms(total)}\n{df.to_string(index=False)}")
    return df


# %% [markdown]
# ### 2.1 Email notifications
#
# Same SMTP contract as the Qwen notebook: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`,
# `SMTP_PASSWORD`, `NOTIFY_EMAIL_TO` from `.env`. A mail failure never kills the run.

# %%
import hashlib
import smtplib
import traceback
from email.message import EmailMessage

RUN_FINGERPRINT = {
    "run_id": hashlib.sha256(
        json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in asdict(CFG).items()}, sort_keys=True, default=str
                   ).encode()).hexdigest()[:12],
}


class Notifier:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.to = os.environ.get("NOTIFY_EMAIL_TO", "").strip()
        self.host = os.environ.get("SMTP_HOST", "").strip()
        self.port = int(os.environ.get("SMTP_PORT", "587") or 587)
        self.user = os.environ.get("SMTP_USER", "").strip()
        self.password = os.environ.get("SMTP_PASSWORD", "").strip()
        self.enabled = bool(cfg.notify_email and self.to and self.host
                            and self.user and self.password)
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
        prefix = "[ROAD-KRAKEN]" + (" FAILURE" if urgent else "")
        line = f"{prefix} {subject}"
        print(f"[notify] {line}")
        if not self.enabled:
            return False
        footer = (f"\n\n--\nrun {RUN_FINGERPRINT['run_id']} | {GPU_NAME} | "
                  f"elapsed {self._elapsed()}\nmodel {self.cfg.model_id}")
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
            return True
        except Exception as exc:
            print(f"[notify] send failed ({type(exc).__name__}: {exc})")
            return False

    def run_started(self) -> None:
        self.send(
            f"kraken run started on {GPU_NAME}",
            f"Fine-tuning {self.cfg.model_id} on folds {list(self.cfg.train_folds)}.\n"
            f"resize {self.cfg.resize} | epochs {self.cfg.epochs} {self.cfg.quit} | "
            f"lr {self.cfg.lr:g} | also_small {self.cfg.also_train_small}\n"
            f"run id {RUN_FINGERPRINT['run_id']}",
        )

    def fold_done(self, tag: str, score: float, cer: float, wer: float) -> None:
        self.send(
            f"fold {tag} OOF {score:.5f}",
            f"tag    {tag}\nscore  {score:.5f}\nCER    {cer:.4f}\nWER    {wer:.4f}",
        )

    def finished(self, oof: float, baseline: float, path: Path) -> None:
        self.send(
            f"kraken run finished - OOF {oof:.5f}",
            f"out-of-fold score  {oof:.5f}\n"
            f"raw CTC identity   {baseline:.5f}  ({oof - baseline:+.5f} from post-processing)\n"
            f"submission         {path}",
        )

    def failed(self, where: str, detail: str) -> None:
        self.send(where, f"The run stopped.\n\n{detail}"[:8000], urgent=True)

    def install_crash_handler(self) -> None:
        def hook(exc_type, exc, tb):
            self.failed("kraken run crashed",
                        "".join(traceback.format_exception(exc_type, exc, tb)))
            sys.__excepthook__(exc_type, exc, tb)
        sys.excepthook = hook


NOTIFY = Notifier(CFG)
NOTIFY.install_crash_handler()
NOTIFY.run_started()

# %% [markdown]
# ### 2.2 Hub and TensorBoard
#
# Same contract as the Qwen notebook. `HF_REPO_ID` + `HF_TOKEN` in `.env` mirror
# fold weights, TensorBoard event files and the two submission CSVs. An empty
# `hf_repo_id` is a no-op. A hub failure never kills training.
#
# ketos itself writes Lightning scalars when `--logger tensorboard` is set; we
# also log OOF score/CER/WER after each fold. View locally with
# `tensorboard --logdir artifacts_kraken/runs --port 6006`, or on the Hub
# Training metrics tab once a `runs/` folder has been pushed.

# %%
class HubSync:
    """Failure-tolerant Hugging Face uploads. Never raises into training."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.repo_id = (cfg.hf_repo_id or "").strip()
        self.api = None
        self.enabled = False
        if not self.repo_id:
            print("[hub] disabled (CFG.hf_repo_id / HF_REPO_ID is empty)")
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

    def _guard(self, what: str, fn, *args, **kwargs):
        if not self.enabled:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            print(f"[hub] FAILED {what}: {type(exc).__name__}: {exc}")
            return None

    def push_folder(self, local: Path, path_in_repo: str, message: str = "") -> bool:
        local = Path(local)
        if not self.enabled or not local.exists():
            return False
        size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file()) / 1e6
        print(f"[hub] uploading {path_in_repo} ({size:.0f} MB)")
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

    def write_card(self, body: str) -> None:
        if not self.enabled:
            return
        card = OUT / "README_hub.md"
        card.write_text(body, encoding="utf-8")
        self.push_file(card, "README.md", "update model card")


class TBLogger:
    root = OUT / "runs"

    def __init__(self, tag: str, enabled: bool = True):
        self.tag, self.writer = tag, None
        self.dir = self.root / tag
        if not enabled:
            return
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.dir))
        except Exception as exc:
            print(f"[tb] disabled ({type(exc).__name__}: {exc})")

    def scalar(self, name: str, value, step: int) -> None:
        if self.writer is None or value is None:
            return
        try:
            self.writer.add_scalar(name, float(value), int(step))
        except Exception:
            pass

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


def ketos_tb_dir(tag: str) -> Path:
    d = TBLogger.root / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_ketos_val_curve(tag: str, log_txt: str) -> None:
    """Scrape ketos validation 'errors / total chars' into TensorBoard."""
    tb = TBLogger(tag, CFG.tensorboard)
    chars, step = None, 0
    for line in log_txt.splitlines():
        m = re.search(r"total chars\s+([\d.]+)", line)
        if m:
            chars = float(m.group(1))
        m = re.search(r"errors\s+([\d.]+)", line)
        if m and chars:
            cer = float(m.group(1)) / max(chars, 1.0)
            tb.scalar("ketos/val_cer", cer, step)
            tb.scalar("ketos/val_acc", 1.0 - cer, step)
            step += 1
    tb.close()


def kraken_model_card(scores: dict | None = None) -> str:
    score_lines = "".join(f"- **{k}**: {v}\n" for k, v in (scores or {}).items())
    return f"""---
license: apache-2.0
base_model: {CFG.model_id}
tags:
  - handwritten-text-recognition
  - ocr
  - kraken
  - ppocrv6
library_name: kraken
---

# R.O.A.D. Barbados — Kraken PP-OCRv6

Fine-tune of [`{CFG.model_id}`](https://huggingface.co/{CFG.model_id}) on the
challenge's own `Train.csv` line crops. Weights only; images are not redistributed.

## Results

{score_lines or "_written when the run completes_"}

- run id: `{RUN_FINGERPRINT.get("run_id")}`
- folds: `{list(CFG.train_folds)}`
- resize: `{CFG.resize}`
- epochs: `{CFG.epochs}` `{CFG.quit}`
- batch / max_width: `{CFG.batch_size}` / `{CFG.max_line_width}`
"""


HUB = HubSync(CFG)
TBLogger.root.mkdir(parents=True, exist_ok=True)
print(f"[tb] {'enabled' if CFG.tensorboard else 'disabled'} -> {TBLogger.root}")
print(f"[tb] tensorboard --logdir {TBLogger.root} --port 6006")
if CFG.hf_repo_id:
    print(f"[tb] hub tab  https://huggingface.co/{CFG.hf_repo_id}/tensorboard")

