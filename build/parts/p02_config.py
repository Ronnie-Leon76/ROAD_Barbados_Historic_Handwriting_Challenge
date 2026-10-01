# %% [markdown]
# ## 2. Configuration
#
# One object controls the whole run. Defaults are auto-scaled to the detected GPU:
# an 8B backbone in bf16 on >= 40 GB cards, a 4B backbone or 4-bit QLoRA on smaller
# ones. Every downstream cell reads from `CFG`, so a single edit here is enough to
# change the experiment. Session knobs go in the `CLI_OVERRIDES` cell immediately
# below; `run_pipeline.py --set` merges on top of them.
#
# Three switches change the shape of the run rather than a hyperparameter:
#
# | Switch | Effect |
# |---|---|
# | `competition_mode` | **Leaderboard recipe**: all five folds + the kraken CTC branch. `--fast` and `--folds` still override it |
# | `finetune_mode` | `"lora"` adapters (default) or `"full"` — every weight trained, with the learning rate, optimiser, batch shape and EMA re-derived to match |
# | `run_grpo` | RL polish on the leaderboard metric. Off by default: with this data the 6 samples/image are nearly identical, so the advantages are noise |
# | `hf_repo_id` | mirrors every checkpoint and artefact to the Hugging Face Hub so the run can be replayed exactly |

# %%
# Session knobs. argparse / `run_pipeline.py --set` merge on top (CLI wins).
# Resolution is commented out on purpose: CACHE_DIR is computed once at import
# from CFG.line_height / max_line_width, so a high-res run needs its own session.
CLI_OVERRIDES = {
    **dict(
        run_grpo=False,
        # --- high-res variant: run in its OWN session (cache dir is fixed at import) ---
        # line_height=192, max_line_width=3840,
        # per_device_batch_size=4, grad_accum=8,   # same effective batch of 32, avoids OOM
        # --- only after the view ablation (Section 15) says dark/light add nothing ---
        # tta_photometric=False,
    ),
    **globals().get("CLI_OVERRIDES", {}),
}

# %%
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
    # One fold is a plumbing check (the 0.905 public score). A leaderboard run
    # needs every fold: that is the largest unused lever in this pipeline.
    train_folds: tuple = (0,)
    competition_mode: bool = True          # True -> all n_folds + CTC branch
                                           # (--fast / CLI --folds still win)
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
    # GRPO optimises the leaderboard metric directly. It is off: on this data the
    # 6 samples per image are nearly identical (reward_std ~0.02–0.04), so the
    # advantages are noise. Re-enable only if the verdict tab shows a real gain.
    run_grpo: bool = False
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
if CFG.competition_mode:
    CFG.train_folds = tuple(range(CFG.n_folds))
    CFG.ctc_branch = True
    print(f"[cfg] COMPETITION MODE: CTC on folds {CFG.train_folds}; VLM runs follow RUN_PLAN")
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

# %% [markdown]
# ### 2.1 Licence guard
#
# The organisers rejected a model purely on licence grounds, so this is enforced in
# code rather than left to memory. The guard reads the licence from the Hugging Face
# model card and refuses anything outside the commercial-use allow-list. If the hub
# is unreachable, it falls back to a hard-coded table of models whose licences were
# verified while writing this notebook.

# %%
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
