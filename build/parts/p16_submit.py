# %% [markdown]
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

# %%
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

# %% [markdown]
# ## 17. Submission
#
# The tuned post-processing is applied to the pooled test N-best lists using a
# language model and lexicon built from the **entire** training set (no fold
# restriction is needed here — the test labels were never part of it).

# %%
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

# %%
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

# %% [markdown]
# ## 18. Where the remaining points are
#
# Ordered by expected gain per GPU-hour after the fold-0 run (best step 532/666,
# post-processing +0.0026, GRPO reward_std ~0.03, EMA ≈ raw).
#
# 1. **Read the view/model ablation table** printed after the post-processing
#    ladder. If "no photometric" matches "ALL views", set `tta_photometric=False`
#    in `CLI_OVERRIDES` and save a third of decode time. If two models beat either
#    one, keep spending on voters rather than on GRPO or rank.
# 2. **More voters, then continue the first adapter.** `RUN_PLAN` trains
#    `fold0_seed7` (same val fold, a real two-model OOF) and `full_from42`
#    (warm-start of fold0_seed42 on every labelled line). Paste the ablation
#    table before buying another seed.
# 3. **Resolution, only if time remains.** `line_height=192` with
#    `max_line_width=3840` needs its *own* session: `CACHE_DIR` is computed once
#    from CFG at import, and `load_line` reads that directory. Drop
#    `per_device_batch_size` to 4 and `grad_accum` to 8 so the effective batch
#    stays 32.
# 4. **Check the oracle gap** printed at the end of `Section 14`. If the oracle over
#    the N-best list is far above the fused score, the ranking is the bottleneck:
#    raise `n_best` and re-run the weight search. If the oracle is close, the
#    *model* is the bottleneck.
# 5. **GRPO is off.** Re-enable it only if a previous verdict tab shows a gain
#    clearly above +0.005. A delta under that on 320 samples is noise, and
#    `accepted = delta > 0` would have treated noise as a win.
# 6. **Do not raise `lora_r`.** Training reward ~0.98 vs held-out ~0.905 is a
#    generalisation gap, not a capacity problem.
# 7. **Read the kraken CTC verdict.** The summary prints the zero-shot and
#    fine-tuned PP-OCRv6 scores and whether the fusion gate kept it. If the
#    fine-tuned CTC top-1 is within a few thousandths of the VLM, it is a genuine
#    second opinion. If it is far behind, it still earns its place through the
#    `c_sim` anchor on the lines where the VLM loops.
# 8. **Longest lines.** Inspect the worst-CER examples printed above: if they cluster
#    at high aspect ratios, they are being squashed by `max_line_width` and deserve a
#    dedicated higher-budget decoding pass.
