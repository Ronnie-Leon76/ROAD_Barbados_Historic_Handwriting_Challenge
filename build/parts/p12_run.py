# %% [markdown]
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

# %%
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

# %%
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

# %% [markdown]
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

# %%
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
