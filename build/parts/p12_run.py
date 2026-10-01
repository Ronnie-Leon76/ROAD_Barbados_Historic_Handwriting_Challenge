# %% [markdown]
# ## 12. Run training
#
# Fold-0 already peaked at step 532 of 666 (val 0.90477, raw). More epochs on that
# same run will not help. We **pull those LoRA weights** from the hub and continue
# from them:
#
# * `fold0_seed42` is restored, never retrained.
# * `full_from42` warm-starts that adapter on every labelled line, at a lower LR,
#   so the extra ~25% of data can move the weights without wiping the first run.
# * `fold0_seed7` is a *new* adapter on the same val fold. Fusion needs a second
#   error pattern; continuing seed-42 a second time would not give it one.
#
# Tags that already exist on the hub or in `runs.json` are skipped.
#
# Budget on a 48 GB card, including decode:
#
# | Tag | Wall clock | Role |
# |---|---|---|
# | `fold0_seed42` | restored | first voter, pulled from the hub |
# | `fold0_seed7` | ~3 h | second voter, same validation fold, from scratch |
# | `full_from42` | ~1.5 h | continue seed-42 on all labelled lines |
#
# High-res (`line_height=192`) is a *different* session: `CACHE_DIR` is fixed at
# import by `load_line`. Do not uncomment those `CLI_OVERRIDES` here.

# %%
import dataclasses

RUN_PLAN = [
    dict(tag="fold0_seed42", fold=0),                          # restored from hub
    dict(tag="fold0_seed7",  fold=0, over=dict(seed=7)),       # 2nd voter, from scratch
    dict(tag="full_from42",  fold=0,                           # continue fold0_seed42
         over=dict(seed=42, epochs=2.0, lr=4e-5),
         init_from="fold0_seed42"),
]

RUN_TRAINING = True     # set False to reuse artifacts/runs.json from a previous pass


def _is_checkpoint_dir(path: Path) -> bool:
    return path.is_dir() and (
        (path / "adapter_config.json").exists()
        or any(path.glob("adapter_model*.safetensors"))
        or any(path.glob("*.safetensors"))
    )


def _fill_val_ids(run: dict) -> dict:
    if run.get("val_ids"):
        return run
    fold = int(run.get("fold", 0))
    try:
        _, va = fold_frames(fold)
        run["val_ids"] = va["ID"].tolist()
    except Exception:
        run["val_ids"] = []
    return run


def _synthesize_run(tag: str, path: Path) -> dict:
    fold = 0
    m = re.match(r"fold(\d+)_", tag)
    if m:
        fold = int(m.group(1))
    run = {
        "tag": tag,
        "fold": fold,
        "best": {"path": str(path), "score": float("nan"), "source": "hub", "step": None},
        "history": pd.DataFrame(),
        "seed": CFG.seed,
        "finetune_mode": CFG.finetune_mode,
    }
    print(f"[hub] synthesized run record for {tag} <- {path}")
    return _fill_val_ids(run)


def restore_runs_from_hub() -> list:
    """Pull fold0_seed42 (and any later voters) so we can continue instead of retrain."""
    local = OUT / "hub_restore"
    root = HUB.pull(["runs.json", "checkpoints/**", "history/**"], local)
    if root is None:
        print("[hub] nothing restored - falling back to local artefacts")
        return []
    root = Path(root)
    runs: list = []
    runs_path = root / "runs.json"
    if runs_path.exists():
        with open(runs_path) as fh:
            runs = json.load(fh)
        for r in runs:
            tag = r["tag"]
            for key, remote in (("best", f"checkpoints/{tag}"),
                                ("grpo", f"checkpoints/grpo_fold{r.get('fold', 0)}")):
                path = root / remote
                if not path.exists():
                    continue
                if key == "best":
                    r.setdefault("best", {})["path"] = str(path)
                else:
                    r["grpo_path"] = str(path)
            hist = root / "history" / f"val_history_{tag}.csv"
            r["history"] = pd.read_csv(hist) if hist.exists() else pd.DataFrame()
            _fill_val_ids(r)
    known = {r["tag"] for r in runs}
    ckpt_root = root / "checkpoints"
    if ckpt_root.exists():
        for d in sorted(ckpt_root.iterdir()):
            if d.name in known or not _is_checkpoint_dir(d):
                continue
            runs.append(_synthesize_run(d.name, d))
    if not runs:
        print("[hub] snapshot had no usable checkpoints - falling back to local artefacts")
        return []
    print(f"[hub] restored {len(runs)} run(s) from {HUB.repo_id}: "
          + ", ".join(r["tag"] for r in runs))
    return runs


def _load_local_runs() -> list:
    path = OUT / "runs.json"
    if not path.exists():
        return []
    with open(path) as fh:
        runs = json.load(fh)
    print(f"[runs] restored {len(runs)} runs from disk")
    return runs


RUNS = restore_runs_from_hub() if CFG.hf_resume else []
if not RUNS:
    RUNS = _load_local_runs()
_done = {r["tag"] for r in RUNS}


def _save_runs():
    with open(OUT / "runs.json", "w") as fh:
        json.dump([{k: v for k, v in r.items() if k != "history"} for r in RUNS], fh, indent=2)
    HUB.push_file(OUT / "runs.json", "runs.json", "record runs")


if RUN_TRAINING:
    stage(f"training {len(RUN_PLAN)} run(s)")
    if any(spec["tag"] not in _done for spec in RUN_PLAN):
        NOTIFY.run_started()
    for spec in RUN_PLAN:
        if spec["tag"] in _done:
            print(f"[runs] {spec['tag']} already trained - skipping")
            continue
        cfg = dataclasses.replace(CFG, **spec.get("over", {}))
        kw = {}
        if spec["tag"].startswith("full_"):
            # Train on every labelled line. The 64-line "val" is only a sanity check;
            # it sits inside the training set, so checkpoint selection is meaningless here.
            kw = dict(train_override=train_df[["ID", "Target"]],
                      val_override=train_df.sample(64, random_state=0)[["ID", "Target"]])
        parent = spec.get("init_from")
        if parent:
            src = next((r for r in RUNS if r["tag"] == parent), None)
            path = Path(src["best"]["path"]) if src and src.get("best", {}).get("path") else None
            if path is None or not path.exists():
                raise SystemExit(f"[runs] {spec['tag']} needs {parent} on disk first "
                                 f"(hub resume failed to pull it)")
            kw["adapter_path"] = path
            print(f"[runs] {spec['tag']} continues from {parent} ({path})")
        RUNS.append(train_fold(spec["fold"], cfg=cfg, tag=spec["tag"], **kw))
        _done.add(spec["tag"])
        _save_runs()
elif not RUNS:
    RUNS = _load_local_runs()

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
# This stage is **off** by default. On the fold-0 run the 6 samples per image were
# nearly identical (`reward_std` 0.02–0.04), so the advantages are noise on 320
# validation lines. Re-enable `run_grpo` in `CLI_OVERRIDES` only if you have a
# verdict-tab gain clearly above +0.005. Three guards still apply if you do:
#
# * **A KL leash** (`beta = 0.02`) back to the SFT policy, so it polishes rather than
#   wanders.
# * **A tiny learning rate** (`2e-6`), two orders below the SFT rate.
# * **An acceptance test** with a 0.004 margin (320-sample noise is about ±0.003).
#   If the polished adapter does not beat SFT by that much, it is discarded.
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

    # 320-sample noise is about ±0.003; treating any positive delta as a win
    # accepts noise. Require a margin that is larger than that.
    accepted = delta > 0.004
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
    _todo = [r for r in RUNS if not r["tag"].startswith("full_")]
    _skipped = [r for r in RUNS if r["tag"].startswith("full_")]
    _todo = [grpo_stage(r) for r in progress(_todo, desc="grpo", unit="fold")]
    RUNS = _todo + _skipped
    _kept = sum("grpo_path" in r for r in RUNS)
    print(f"\n[grpo] {_kept}/{len(_todo)} fold models improved and will decode with GRPO weights")
    if _skipped:
        print(f"[grpo] skipped {len(_skipped)} all-data run(s) (leaky val)")
else:
    print("[grpo] disabled (CFG.run_grpo = False)")
