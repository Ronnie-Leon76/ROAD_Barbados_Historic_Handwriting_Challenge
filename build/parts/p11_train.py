# %% [markdown]
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
# A larger rank (`lora_r = 64`) is the wrong next step on this data: training
# reward is already ~0.98 against ~0.905 held-out, which is a generalisation gap.
# More seeds and more labelled lines beat more adapter capacity.

# %%
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
               val_override: pd.DataFrame | None = None,
               adapter_path: str | Path | None = None) -> dict:
    """Train one adapter. `adapter_path` warm-starts from an existing LoRA instead
    of initialising a new one on the base model."""
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
          f" of peak lr {cfg.lr:g}")
    if adapter_path:
        print(f"[train] warm-start from {adapter_path}")
    print("=" * 78)

    tb = TBLogger(tag, enabled=cfg.tensorboard)
    tb.text("config", "```json\n" + json.dumps(
        {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(cfg).items()},
        indent=2) + "\n```")

    model = build_model(cfg, adapter_path=adapter_path, for_training=True)
    args = make_training_args(cfg, OUT / "checkpoints" / tag, steps_per_epoch, tb.dir)

    # warmup_steps=4 -> decay (1+n)/(5+n), a running average over roughly the last
    # quarter of a 666-step run. The class default of 20 never reaches cfg.ema_decay
    # (0.999) in that many steps, which is why raw and EMA scored the same.
    ema = AdapterEMA(model, cfg.ema_decay, warmup_steps=4) if cfg.ema_decay and cfg.ema_decay > 0 else None
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
