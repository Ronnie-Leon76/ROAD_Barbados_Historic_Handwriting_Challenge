# %% [markdown]
# ### 15.2 Feature extraction and weight search
#
# Candidate features, computed once and reused across every search iteration:
#
# | Feature | Meaning |
# |---|---|
# | `s_max`, `s_mean` | best / average beam sequence log-probability for this string |
# | `votes` | how many (beam, TTA view, fold) combinations produced this exact string — the single strongest signal in the table |
# | `lm` | character n-gram log-probability per character, fold-safe |
# | `oov` | fraction of words absent from the fold's training vocabulary |
# | `lenz` | absolute z-score of the length against the training length distribution |
#
# The search is a plain random search over the linear weights against the real
# competition metric. With ~5 free parameters and a few hundred evaluations that is
# both sufficient and much harder to over-fit than a gradient-based fit would be.

# %%
@dataclass
class PostConfig:
    w_score: float = 1.0
    w_votes: float = 0.35
    w_lm: float = 0.15
    w_oov: float = -0.25
    w_lenz: float = -0.02
    use_rover: bool = True
    rover_level: str = "char"
    rover_temp: float = 0.08
    repair: bool = False
    repair_sim: float = 90.0
    repair_freq: int = 3
    restrict_charset: bool = True
    collapse: bool = True


FOLD_LM: dict[int, CharLM] = {}
FOLD_LEX: dict[int, Lexicon] = {}


def fold_resources(fold: int) -> tuple[CharLM, Lexicon]:
    if fold not in FOLD_LM:
        texts = train_df.loc[train_df.fold != fold, "Target"].tolist() if fold >= 0 \
            else train_df["Target"].tolist()
        FOLD_LM[fold] = CharLM(texts, order=CFG.lm_order)
        FOLD_LEX[fold] = Lexicon(texts)
    return FOLD_LM[fold], FOLD_LEX[fold]


def clean_candidate(text: str, restrict: bool, collapse: bool) -> str:
    text = normalize_text(text)
    if collapse:
        text = collapse_repeats(text)
    if restrict:
        text = normalize_text(restrict_charset(text))
    return text


def build_features(nbest: dict, ids: Sequence[str], fold_of: dict,
                   restrict: bool = True, collapse: bool = True) -> dict:
    """id -> {'texts': [...], 'M': (n_cand, 5) feature matrix}"""
    feats = {}
    for image_id in progress(ids, desc="features", unit="line"):
        lm, lex = fold_resources(fold_of.get(image_id, -1))
        agg: dict[str, dict] = {}
        for c in nbest.get(image_id, []):
            text = clean_candidate(c["text"], restrict, collapse)
            rec = agg.setdefault(text, {"scores": [], "votes": 0.0})
            rec["scores"].append(float(c["score"]))
            rec["votes"] += 1.0
        if not agg:
            agg[""] = {"scores": [-50.0], "votes": 1.0}
        texts, rows = [], []
        for text, rec in agg.items():
            s = np.asarray(rec["scores"], dtype=np.float64)
            rows.append([
                float(s.max()),
                float(s.mean()),
                math.log1p(rec["votes"]),
                lm.logprob_per_char(text),
                lex.oov_rate(text),
                abs((len(text) - lex.mean_chars) / lex.std_chars),
            ])
            texts.append(text)
        feats[image_id] = {"texts": texts, "M": np.asarray(rows, dtype=np.float64)}
    return feats


def combined_scores(M: np.ndarray, pc: PostConfig) -> np.ndarray:
    return (pc.w_score * (0.5 * M[:, 0] + 0.5 * M[:, 1])
            + pc.w_votes * M[:, 2]
            + pc.w_lm * M[:, 3]
            + pc.w_oov * M[:, 4]
            + pc.w_lenz * M[:, 5])


def apply_post(feats: dict, ids: Sequence[str], pc: PostConfig,
               lexicon_for: Callable[[str], Lexicon] | None = None) -> dict:
    out = {}
    for image_id in ids:
        f = feats[image_id]
        texts, M = f["texts"], f["M"]
        s = combined_scores(M, pc)
        if pc.use_rover and len(texts) > 1:
            z = s - s.max()
            w = np.exp(z / max(pc.rover_temp, 1e-3))
            w = w / w.sum()
            keep = w > 0.01
            if keep.sum() >= 2:
                pred = rover_fuse([t for t, k in zip(texts, keep) if k],
                                  w[keep].tolist(), level=pc.rover_level)
            else:
                pred = texts[int(np.argmax(s))]
        else:
            pred = texts[int(np.argmax(s))]
        if not pred.strip():
            order = np.argsort(-s)
            for j in order:
                if texts[j].strip():
                    pred = texts[j]
                    break
        if pc.repair and lexicon_for is not None:
            pred = repair_words(pred, lexicon_for(image_id),
                                min_sim=pc.repair_sim, min_freq=pc.repair_freq)
        out[image_id] = normalize_text(pred)
    return out


stage("post-processing weight search")
FOLD_OF = dict(zip(train_df.ID, train_df.fold))
OOF_FEATS = build_features(OOF_NBEST, OOF_IDS, FOLD_OF,
                           restrict=True, collapse=True)


def oof_score(pc: PostConfig, feats: dict | None = None) -> float:
    preds = apply_post(feats if feats is not None else OOF_FEATS, OOF_IDS, pc,
                       lexicon_for=lambda i: fold_resources(FOLD_OF.get(i, -1))[1])
    return competition_score(OOF_REFS, [preds[i] for i in OOF_IDS])["score"]


# ---------------------------------------------------------------- stage A --
print("\n[tune] stage A: rescoring weights (ROVER off)")
base_pc = PostConfig(use_rover=False, repair=False)
best_pc, best_val = base_pc, oof_score(base_pc)
print(f"  default weights            {best_val:.5f}")

rng = np.random.default_rng(0)
t0 = time.time()
_bar = progress(range(CFG.lm_tune_iters), desc="tune A", unit="trial")
for it in _bar:
    trial = PostConfig(
        w_score=float(rng.uniform(0.0, 2.0)),
        w_votes=float(rng.uniform(0.0, 1.5)),
        w_lm=float(rng.uniform(0.0, 1.2)),
        w_oov=float(rng.uniform(-1.5, 0.1)),
        w_lenz=float(rng.uniform(-0.3, 0.05)),
        use_rover=False, repair=False,
    )
    val = oof_score(trial)
    if val > best_val:
        best_pc, best_val = trial, val
        _bar.write(f"  it {it:4d}  {val:.5f}  "
                   f"score={trial.w_score:.2f} votes={trial.w_votes:.2f} "
                   f"lm={trial.w_lm:.2f} oov={trial.w_oov:.2f} lenz={trial.w_lenz:.3f}")
    _bar.set_postfix_str(f"best {best_val:.5f}", refresh=False)
_bar.close()
print(f"[tune] stage A best {best_val:.5f} in {time.time()-t0:.0f}s")

# ---------------------------------------------------------------- stage B --
print("\n[tune] stage B: ROVER")
stage_b = [(best_pc, best_val, "off")]
for level in ("char", "word"):
    for temp in (0.03, 0.08, 0.2, 0.5):
        trial = PostConfig(**{**asdict(best_pc), "use_rover": True,
                              "rover_level": level, "rover_temp": temp})
        val = oof_score(trial)
        stage_b.append((trial, val, f"{level} T={temp}"))
        print(f"  rover {level:<5} T={temp:<5} {val:.5f}")
best_pc, best_val, chosen = max(stage_b, key=lambda kv: kv[1])
print(f"[tune] stage B best: {chosen} -> {best_val:.5f}")

# ---------------------------------------------------------------- stage C --
print("\n[tune] stage C: lexicon repair")
stage_c = [(best_pc, best_val, "off")]
for sim in (86.0, 90.0, 94.0):
    for freq in (2, 5, 15):
        trial = PostConfig(**{**asdict(best_pc), "repair": True,
                              "repair_sim": sim, "repair_freq": freq})
        val = oof_score(trial)
        stage_c.append((trial, val, f"sim={sim} freq={freq}"))
        print(f"  repair sim={sim} freq={freq:<3} {val:.5f}")
best_pc, best_val, chosen = max(stage_c, key=lambda kv: kv[1])
print(f"[tune] stage C best: {chosen} -> {best_val:.5f}")

# ---------------------------------------------------------------- stage D --
print("\n[tune] stage D: charset restriction ablation")
_unrestricted = build_features(OOF_NBEST, OOF_IDS, FOLD_OF, restrict=False, collapse=True)
for restrict, feats in ((True, OOF_FEATS), (False, _unrestricted)):
    trial = PostConfig(**{**asdict(best_pc), "restrict_charset": restrict})
    val = oof_score(trial, feats=feats)
    print(f"  restrict_charset={restrict}  {val:.5f}")
    if not restrict and val > best_val:
        best_pc, best_val = trial, val
        print("  keeping the unrestricted charset")

POST = best_pc
with open(OUT / "post_config.json", "w") as fh:
    json.dump(asdict(POST), fh, indent=2)
print(f"\n[tune] final post-processing config:\n{json.dumps(asdict(POST), indent=2)}")

# %%
# Incremental contribution of each stage, measured out-of-fold.
ladder = [
    ("raw best beam", PostConfig(w_score=1.0, w_votes=0.0, w_lm=0.0, w_oov=0.0,
                                 w_lenz=0.0, use_rover=False, repair=False,
                                 restrict_charset=False, collapse=False)),
    ("+ cleanup (charset, loops)", PostConfig(w_score=1.0, w_votes=0.0, w_lm=0.0,
                                              w_oov=0.0, w_lenz=0.0, use_rover=False,
                                              repair=False)),
    ("+ vote/LM rescoring", PostConfig(**{**asdict(POST), "use_rover": False, "repair": False})),
    ("+ ROVER", PostConfig(**{**asdict(POST), "repair": False})),
    ("+ lexicon repair", POST),
]
rows = []
prev = None
for name, pc in ladder:
    val = oof_score(pc)
    rows.append({"stage": name, "oof_score": round(val, 5),
                 "delta": None if prev is None else round(val - prev, 5)})
    prev = val
LADDER = pd.DataFrame(rows)
print(LADDER.to_string(index=False))
LADDER.to_csv(OUT / "postprocessing_ladder.csv", index=False)

# %%
# Error inspection: the worst out-of-fold lines, to decide where to spend next.
_final_oof = apply_post(OOF_FEATS, OOF_IDS, POST,
                        lexicon_for=lambda i: fold_resources(FOLD_OF.get(i, -1))[1])
_err = error_rates(OOF_REFS, [_final_oof[i] for i in OOF_IDS], "char")["per_sample"]
_worst_idx = np.argsort(-_err)[:12]
print("worst out-of-fold lines\n" + "-" * 78)
for j in _worst_idx:
    print(f"CER {_err[j]:.2f}  id={OOF_IDS[j]}")
    print(f"  ref  {OOF_REFS[j]}")
    print(f"  pred {_final_oof[OOF_IDS[j]]}")
print("-" * 78)
print(f"lines with CER = 0     : {(_err == 0).mean():.1%}")
print(f"lines with CER > 0.5   : {(_err > 0.5).mean():.1%}")
