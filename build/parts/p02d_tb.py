# %% [markdown]
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

# %%
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
