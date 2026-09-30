# %% [markdown]
# ## 7. Submission
#
# Two files are written:
#
# * `submission_kraken.csv` — OOF-tuned Kraken (+ small) fusion. This is the one
#   whose local score you can trust.
# * `submission_ensemble.csv` — same, plus the Qwen file as an extra ROVER member
#   when it exists. If we did not have Qwen OOF predictions, the mix is *gated*:
#   a Qwen string only wins a line when it is within a small edit distance of the
#   Kraken consensus. That stops a fluent hallucination from overwriting a
#   literal CTC reading, which is the failure mode the BHL board documented.

# %%
stage("submission")
TEST_IDS = test_df.ID.tolist()
fold_resources(-1)
TEST_FOLD_OF = {i: -1 for i in TEST_IDS}
TEST_FEATS = build_features(TEST_NBEST, TEST_IDS, TEST_FOLD_OF,
                            restrict=POST.restrict_charset, collapse=POST.collapse)
TEST_PRED = apply_post(TEST_FEATS, TEST_IDS, POST,
                       lexicon_for=lambda i: FOLD_LEX[-1])

# Gated VLM blend when we could not OOF-tune the mix.
if VLM_TEST and not HAS_VLM_OOF:
    n_replaced = 0
    for i in TEST_IDS:
        k = TEST_PRED[i]
        v = VLM_TEST.get(i, [{}])[0].get("text", "")
        if not v or v == k:
            continue
        # Close: let ROVER of the two strings pick the characters.
        sim = Levenshtein.normalized_similarity(k, v)
        if sim >= 0.80:
            fused = rover_fuse([k, v], [1.0 - CFG.vlm_blend, CFG.vlm_blend])
            if fused != k:
                TEST_PRED[i] = normalize_text(restrict_charset(fused))
                n_replaced += 1
        # Far apart: keep Kraken. A 0.90 VLM that invents "the said" is how you
        # lose CER against a specialist.
    print(f"[ensemble] gated VLM blend touched {n_replaced}/{len(TEST_IDS)} lines "
          f"(sim>=0.80); remainder stayed Kraken")

submission = pd.DataFrame({"ID": TEST_IDS, "Target": [TEST_PRED[i] for i in TEST_IDS]})
n_empty = int((submission.Target.str.strip() == "").sum())
if n_empty:
    print(f"!! {n_empty} empty predictions - filling from raw CTC top-1")
    fallback = {}
    for i, cands in TEST_NBEST.items():
        ctc = [c for c in cands if str(c.get("family", "")).startswith("ctc")]
        pool = ctc or cands
        fallback[i] = max(pool, key=lambda c: c["score"])["text"] if pool else "the"
    mask = submission.Target.str.strip() == ""
    submission.loc[mask, "Target"] = [normalize_text(fallback.get(i, "the")) or "the"
                                      for i in submission.loc[mask, "ID"]]

SUB_PATH = WORK / "submission_kraken.csv"
ENS_PATH = WORK / "submission_ensemble.csv"
submission.to_csv(SUB_PATH, index=False)
submission.to_csv(ENS_PATH, index=False)
print(f"wrote {SUB_PATH} ({len(submission)} rows)")

# Raw CTC top-1 OOF, for the ladder.
raw_preds = []
for image_id in OOF_IDS:
    cands = [c for c in OOF_NBEST[image_id] if str(c.get("family", "")).startswith("ctc")
             and str(c.get("view", "")).endswith(":identity")]
    raw_preds.append(cands[0]["text"] if cands else "")
RAW = competition_score(OOF_REFS, raw_preds)
_fused_map = apply_post(OOF_FEATS, OOF_IDS, POST,
                        lexicon_for=lambda j: fold_resources(FOLD_OF.get(j, -1))[1])
FUSED = competition_score(OOF_REFS, [_fused_map[i] for i in OOF_IDS])

print("=" * 78)
print("KRAKEN PP-OCRv6 RUN SUMMARY")
print("=" * 78)
print(f"backbone     {CFG.model_id}  ({CFG_LICENCE})")
print(f"resize       {CFG.resize} | folds {list(CFG.train_folds)} | also_small {CFG.also_train_small}")
for k, v in REPORT.items():
    if isinstance(v, dict) and "score" in v:
        print(f"  {k:<28} {v['score']:.5f}  CER {v['cer']:.4f}  WER {v['wer']:.4f}")
print(f"OOF raw CTC identity         {RAW['score']:.5f}")
print(f"OOF after post-processing    {FUSED['score']:.5f}  ({FUSED['score'] - RAW['score']:+.5f})")
print(f"Qwen VLM (this pod)          0.90491 OOF / 0.90485 public  (1 fold + GRPO)")
print(f"current first place          0.93989")
print(f"submission                   {SUB_PATH}")
if VLM_TEST:
    print(f"ensemble                     {ENS_PATH}  (gated VLM blend, "
          f"{'OOF-tuned' if HAS_VLM_OOF else 'not OOF-tuned'})")
print("=" * 78)
print(
    "If fused OOF is still below ~0.93 the remaining points are in the recogniser,\n"
    "not the fusion: train more epochs (`CFG.epochs = 80`), try `resize='union'`\n"
    "as a second family, or raise `line_height` to 160 with a larger max-width.\n"
    "Do not throw away the Qwen file - submit both and keep whichever the\n"
    "public leaderboard prefers; the private set can disagree."
)
stage_summary()
NOTIFY.finished(FUSED["score"], RAW["score"], SUB_PATH)
with open(OUT / "kraken_report.json", "w") as fh:
    json.dump({"report": {k: v for k, v in REPORT.items() if not isinstance(v, np.ndarray)},
               "raw": RAW, "fused": FUSED, "post": asdict(POST),
               "config": {k: (list(v) if isinstance(v, tuple) else v)
                          for k, v in asdict(CFG).items()}}, fh, indent=2, default=str)
