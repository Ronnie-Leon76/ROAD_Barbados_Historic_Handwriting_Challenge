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
# | `c_score`, `c_votes` | the same confidence / agreement signals from the kraken CTC member (Section 14.1) |
# | `c_sim` | similarity of the string to the CTC consensus reading — the anchor that penalises VLM output drifting away from the page |
#
# Scores from the two recognisers live on different scales, so they get separate
# columns; a string one family never produced gets that family's per-line floor.
# The whole search runs twice — VLM-only and VLM+CTC — and the CTC member is kept
# only if it wins out-of-fold.
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
    # kraken CTC member (Section 14.1); all zero == the VLM-only behaviour.
    use_ctc: bool = False
    w_ctc_score: float = 0.0
    w_ctc_votes: float = 0.0
    w_ctc_sim: float = 0.0

    @property
    def families(self) -> tuple:
        return ("vlm", "ctc") if self.use_ctc else ("vlm",)


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


def family_of(cand: dict) -> str:
    return cand.get("family") or "vlm"


def build_features(nbest: dict, ids: Sequence[str], fold_of: dict,
                   restrict: bool = True, collapse: bool = True,
                   families: Sequence[str] = ("vlm", "ctc")) -> dict:
    """id -> {'texts': [...], 'M': (n_cand, 9) feature matrix}"""
    feats = {}
    for image_id in progress(ids, desc="features", unit="line"):
        lm, lex = fold_resources(fold_of.get(image_id, -1))
        agg: dict[str, dict] = {}
        for c in nbest.get(image_id, []):
            fam = family_of(c)
            if fam not in families:
                continue
            text = clean_candidate(c["text"], restrict, collapse)
            rec = agg.setdefault(text, {"vlm": [], "ctc": []})
            rec[fam].append(float(c["score"]))
        if not agg:
            agg[""] = {"vlm": [-50.0], "ctc": []}

        v_all = [x for r in agg.values() for x in r["vlm"]]
        c_all = [x for r in agg.values() for x in r["ctc"]]
        v_floor = (min(v_all) - 1.0) if v_all else 0.0
        c_floor = (min(c_all) - 1.0) if c_all else 0.0
        consensus = None
        if c_all:
            consensus = max((t for t, r in agg.items() if r["ctc"]),
                            key=lambda t: (len(agg[t]["ctc"]), max(agg[t]["ctc"])))

        texts, rows = [], []
        for text, rec in agg.items():
            v = np.asarray(rec["vlm"], dtype=np.float64)
            c = np.asarray(rec["ctc"], dtype=np.float64)
            rows.append([
                float(v.max()) if v.size else v_floor,
                float(v.mean()) if v.size else v_floor,
                math.log1p(v.size),
                lm.logprob_per_char(text),
                lex.oov_rate(text),
                abs((len(text) - lex.mean_chars) / lex.std_chars),
                float(c.max()) if c.size else c_floor,
                math.log1p(c.size),
                float(Levenshtein.normalized_similarity(text, consensus))
                if consensus is not None else 0.0,
            ])
            texts.append(text)
        feats[image_id] = {"texts": texts, "M": np.asarray(rows, dtype=np.float64)}
    return feats


def combined_scores(M: np.ndarray, pc: PostConfig) -> np.ndarray:
    return (pc.w_score * (0.5 * M[:, 0] + 0.5 * M[:, 1])
            + pc.w_votes * M[:, 2]
            + pc.w_lm * M[:, 3]
            + pc.w_oov * M[:, 4]
            + pc.w_lenz * M[:, 5]
            + pc.w_ctc_score * M[:, 6]
            + pc.w_ctc_votes * M[:, 7]
            + pc.w_ctc_sim * M[:, 8])


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
HAS_CTC = bool(globals().get("HAS_CTC", False))


def oof_score(pc: PostConfig, feats: dict | None = None) -> float:
    preds = apply_post(feats if feats is not None else OOF_FEATS, OOF_IDS, pc,
                       lexicon_for=lambda i: fold_resources(FOLD_OF.get(i, -1))[1])
    return competition_score(OOF_REFS, [preds[i] for i in OOF_IDS])["score"]


def tune_post(use_ctc: bool, label: str) -> tuple:
    """Stages A-D for one member set. Returns (PostConfig, oof score, features)."""
    fams = ("vlm", "ctc") if use_ctc else ("vlm",)
    feats = build_features(OOF_NBEST, OOF_IDS, FOLD_OF, restrict=True, collapse=True,
                           families=fams)
    print(f"\n{'=' * 30} tuning: {label} {'=' * 30}")

    # ------------------------------------------------------------ stage A --
    print("[tune] stage A: rescoring weights (ROVER off)")
    base_pc = PostConfig(use_rover=False, repair=False, use_ctc=use_ctc)
    starts = [("default weights", base_pc)]
    if use_ctc:
        # Two hand-picked starting points so the search does not have to find them:
        # the VLM defaults anchored to the CTC reading, and CTC-only selection.
        starts += [
            ("defaults + CTC anchor", PostConfig(**{**asdict(base_pc), "w_ctc_sim": 1.0,
                                                    "w_ctc_votes": 0.3})),
            ("CTC only", PostConfig(w_score=0.0, w_votes=0.0, w_lm=0.1, w_oov=0.0,
                                    w_lenz=0.0, use_rover=False, repair=False,
                                    use_ctc=True, w_ctc_score=1.0, w_ctc_votes=1.0)),
        ]
    best_pc, best_val = None, -1.0
    for name, pc in starts:
        val = oof_score(pc, feats)
        print(f"  {name:<26} {val:.5f}")
        if val > best_val:
            best_pc, best_val = pc, val

    rng = np.random.default_rng(0)
    t0 = time.time()
    _bar = progress(range(CFG.lm_tune_iters), desc=f"tune A {label}", unit="trial")
    for it in _bar:
        trial = PostConfig(
            w_score=float(rng.uniform(0.0, 2.0)),
            w_votes=float(rng.uniform(0.0, 1.5)),
            w_lm=float(rng.uniform(0.0, 1.2)),
            w_oov=float(rng.uniform(-1.5, 0.1)),
            w_lenz=float(rng.uniform(-0.3, 0.05)),
            use_rover=False, repair=False, use_ctc=use_ctc,
        )
        if use_ctc:
            trial.w_ctc_score = float(rng.uniform(0.0, 1.5))
            trial.w_ctc_votes = float(rng.uniform(0.0, 1.5))
            trial.w_ctc_sim = float(rng.uniform(0.0, 4.0))
        val = oof_score(trial, feats)
        if val > best_val:
            best_pc, best_val = trial, val
            _bar.write(f"  it {it:4d}  {val:.5f}  "
                       f"score={trial.w_score:.2f} votes={trial.w_votes:.2f} "
                       f"lm={trial.w_lm:.2f} oov={trial.w_oov:.2f} lenz={trial.w_lenz:.3f}"
                       + (f" | ctc score={trial.w_ctc_score:.2f} votes={trial.w_ctc_votes:.2f}"
                          f" sim={trial.w_ctc_sim:.2f}" if use_ctc else ""))
        _bar.set_postfix_str(f"best {best_val:.5f}", refresh=False)
    _bar.close()
    print(f"[tune] stage A best {best_val:.5f} in {time.time()-t0:.0f}s")

    # ------------------------------------------------------------ stage B --
    print("[tune] stage B: ROVER")
    stage_b = [(best_pc, best_val, "off")]
    for level in ("char", "word"):
        for temp in (0.03, 0.08, 0.2, 0.5):
            trial = PostConfig(**{**asdict(best_pc), "use_rover": True,
                                  "rover_level": level, "rover_temp": temp})
            val = oof_score(trial, feats)
            stage_b.append((trial, val, f"{level} T={temp}"))
            print(f"  rover {level:<5} T={temp:<5} {val:.5f}")
    best_pc, best_val, chosen = max(stage_b, key=lambda kv: kv[1])
    print(f"[tune] stage B best: {chosen} -> {best_val:.5f}")

    # ------------------------------------------------------------ stage C --
    print("[tune] stage C: lexicon repair")
    stage_c = [(best_pc, best_val, "off")]
    for sim in (86.0, 90.0, 94.0):
        for freq in (2, 5, 15):
            trial = PostConfig(**{**asdict(best_pc), "repair": True,
                                  "repair_sim": sim, "repair_freq": freq})
            val = oof_score(trial, feats)
            stage_c.append((trial, val, f"sim={sim} freq={freq}"))
            print(f"  repair sim={sim} freq={freq:<3} {val:.5f}")
    best_pc, best_val, chosen = max(stage_c, key=lambda kv: kv[1])
    print(f"[tune] stage C best: {chosen} -> {best_val:.5f}")

    # ------------------------------------------------------------ stage D --
    print("[tune] stage D: charset restriction ablation")
    unrestricted = build_features(OOF_NBEST, OOF_IDS, FOLD_OF, restrict=False,
                                  collapse=True, families=fams)
    trial = PostConfig(**{**asdict(best_pc), "restrict_charset": False})
    val = oof_score(trial, unrestricted)
    print(f"  restrict_charset=True   {best_val:.5f}")
    print(f"  restrict_charset=False  {val:.5f}")
    if val > best_val:
        best_pc, best_val, feats = trial, val, unrestricted
        print("  keeping the unrestricted charset")
    return best_pc, best_val, feats


TUNED = {"vlm": tune_post(False, "VLM only")}
if HAS_CTC:
    TUNED["vlm+ctc"] = tune_post(True, "VLM + kraken CTC")

print("\n[tune] member-set gate (out-of-fold):")
for _k, (_pc, _v, _) in TUNED.items():
    print(f"  {_k:<10} {_v:.5f}")
_choice = max(TUNED, key=lambda k: TUNED[k][1])
POST, best_val, OOF_FEATS = TUNED[_choice]
if HAS_CTC:
    _delta = TUNED["vlm+ctc"][1] - TUNED["vlm"][1]
    verdict = "KEPT" if _choice == "vlm+ctc" else "DROPPED"
    print(f"[tune] kraken CTC member {verdict} ({_delta:+.5f} OOF)")
    CTC_REPORT["gate_delta"] = _delta
    CTC_REPORT["gate_kept"] = _choice == "vlm+ctc"
    # All three weights are fitted on the same OOF rows the gate reads, so a
    # delta within ~1e-4 is not evidence either way.
    if abs(_delta) < 1e-4:
        print("[tune] note: the difference is within tuning noise")

with open(OUT / "post_config.json", "w") as fh:
    json.dump(asdict(POST), fh, indent=2)
print(f"\n[tune] final post-processing config:\n{json.dumps(asdict(POST), indent=2)}")

# %%
# Incremental contribution of each stage, measured out-of-fold.
_vlm_feats = TUNED["vlm"][2]
_vlm_post = TUNED["vlm"][0]
ladder = [
    ("raw best beam", _vlm_feats, PostConfig(w_score=1.0, w_votes=0.0, w_lm=0.0,
                                             w_oov=0.0, w_lenz=0.0, use_rover=False,
                                             repair=False, restrict_charset=False,
                                             collapse=False)),
    ("+ cleanup (charset, loops)", _vlm_feats, PostConfig(w_score=1.0, w_votes=0.0,
                                                          w_lm=0.0, w_oov=0.0,
                                                          w_lenz=0.0, use_rover=False,
                                                          repair=False)),
    ("+ vote/LM rescoring", _vlm_feats,
     PostConfig(**{**asdict(_vlm_post), "use_rover": False, "repair": False})),
    ("+ ROVER", _vlm_feats, PostConfig(**{**asdict(_vlm_post), "repair": False})),
    ("+ lexicon repair", _vlm_feats, _vlm_post),
]
if HAS_CTC:
    ladder.append(("+ kraken CTC member" + ("" if POST.use_ctc else " (rejected)"),
                   TUNED["vlm+ctc"][2], TUNED["vlm+ctc"][0]))
rows = []
prev = None
for name, feats_, pc in ladder:
    val = oof_score(pc, feats_)
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
