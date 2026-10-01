# %% [markdown]
# ## 14. Out-of-fold and test decoding
#
# Each trained adapter is loaded **once** and used to decode both its own validation
# fold and the full test set, so the base model is materialised as few times as
# possible. Every decode produces an N-best list with beam scores across all TTA
# views; those pooled lists are the input to the fusion stage.
#
# Cost control: the number of forward passes is
# `models x (len(scales) + 2 if photometric) x beams`. On a 48 GB card, one adapter
# over the 1 374 test images with 4 views and 5 beams takes roughly an hour. Set
# `FAST_DECODE = True` for a single greedy view while iterating. Predictions are
# pulled from / pushed to the hub so a dead pod does not redo 1–2 hours per adapter.

# %%
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
    HUB.push_file(oof_path, f"preds/{oof_path.name}", f"decoded {tag} oof")
    HUB.push_file(test_path, f"preds/{test_path.name}", f"decoded {tag} test")

    del engine, model
    free_gpu()

    refs = train_df.set_index("ID").loc[val_ids, "Target"].tolist()
    hyps = [top1(oof)[i] for i in val_ids]
    res = competition_score(refs, hyps)
    print(f"[decode] {tag}: raw top-1 OOF score {res['score']:.5f} "
          f"(CER {res['cer']:.4f}, WER {res['wer']:.4f})")
    return {"tag": tag, "oof": oof, "test": test}


stage("decode out-of-fold + test")
HUB.pull(["preds/**"], OUT)
DECODES = [decode_run(r) for r in progress(RUNS, desc="adapters", unit="model")]
HUB.push_folder(OUT / "preds", "preds", "decoded n-best lists")

# %%
# Assemble the out-of-fold table. All-data models are excluded: their "validation"
# lines sat inside the training set, so they would leak into the fusion search.
# They still vote on TEST_NBEST below.
OOF_NBEST: dict[str, list] = defaultdict(list)
for d in DECODES:
    if d["tag"].startswith("full_"):     # its "validation" lines were in its training set
        continue
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

# %%
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
