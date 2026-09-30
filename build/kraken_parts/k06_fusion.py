# %% [markdown]
# ## 6. Fusion
#
# Each fold × TTA view is a hypothesis. Character ROVER repairs the single-glyph
# disagreements that dominate CER; a fold-safe character LM and a frequency
# lexicon catch the word errors that dominate WER (the VLM's real gap to first
# place). If a Qwen `submission.csv` is on disk it becomes another ROVER member.
# The mix is only OOF-tuned when `CFG.vlm_oof_csv` exists; otherwise a gated
# blend (prefer Kraken unless the two strings are close) is used so a bad VLM
# line cannot overwrite a good CTC one.

# %%
class CharLM:
    def __init__(self, texts: Sequence[str], order: int = 7, alpha: float = 0.4):
        self.order, self.alpha = order, alpha
        self.ctx: dict[str, Counter] = defaultdict(Counter)
        self.ctx_total: Counter = Counter()
        self.uni: Counter = Counter()
        bos = "\x02"
        for t in texts:
            s = bos * (order - 1) + t + "\x03"
            for i in range(order - 1, len(s)):
                ch = s[i]
                self.uni[ch] += 1
                for k in range(order):
                    h = s[i - k:i]
                    self.ctx[h][ch] += 1
                    self.ctx_total[h] += 1
        self.n_uni = sum(self.uni.values())
        self.v_uni = len(self.uni) + 1

    def _logp(self, ctx: str, ch: str) -> float:
        penalty = 0.0
        for k in range(len(ctx), -1, -1):
            h = ctx[len(ctx) - k:] if k else ""
            c = self.ctx[h].get(ch, 0) if h in self.ctx else 0
            if c > 0:
                return math.log(c / self.ctx_total[h]) + penalty
            penalty += math.log(self.alpha)
        return math.log((self.uni.get(ch, 0) + 1) / (self.n_uni + self.v_uni)) + penalty

    def logprob(self, text: str) -> float:
        s = "\x02" * (self.order - 1) + text + "\x03"
        return sum(self._logp(s[i - self.order + 1:i], s[i])
                   for i in range(self.order - 1, len(s)))

    def logprob_per_char(self, text: str) -> float:
        return self.logprob(text) / max(len(text) + 1, 1)


class Lexicon:
    def __init__(self, texts: Sequence[str]):
        self.freq = Counter(w for t in texts for w in t.split())
        self.vocab = set(self.freq)
        self.lower: dict[str, list[str]] = defaultdict(list)
        for w in self.vocab:
            self.lower[w.lower()].append(w)
        self.choices = sorted(self.vocab)
        self.mean_chars = float(np.mean([len(t) for t in texts])) if len(texts) else 60.0
        self.std_chars = float(np.std([len(t) for t in texts])) or 1.0

    def oov_rate(self, text: str) -> float:
        words = text.split()
        return (sum(w not in self.vocab for w in words) / len(words)) if words else 1.0


def match_case(source: str, target: str) -> str:
    if source.isupper() and len(source) > 1:
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def repair_words(text: str, lex: Lexicon, min_sim: float = 88.0,
                 min_freq: int = 3, max_len_ratio: float = 0.34) -> str:
    out = []
    for word in text.split():
        core = word.strip(".,;:()\"'")
        if not core or core in lex.vocab or len(core) < 3:
            out.append(word)
            continue
        if lex.lower.get(core.lower()):
            out.append(word)
            continue
        budget = max(1, int(len(core) * max_len_ratio))
        cands = rf_process.extract(
            core, lex.choices, scorer=Levenshtein.normalized_similarity,
            processor=str.lower, score_cutoff=min_sim / 100.0, limit=8)
        viable = [(sim + 0.02 * math.log10(lex.freq[c] + 1), c)
                  for c, sim, _ in cands
                  if lex.freq[c] >= min_freq
                  and Levenshtein.distance(core.lower(), c.lower()) <= budget]
        if not viable:
            out.append(word)
            continue
        best = max(viable)[1]
        out.append(word.replace(core, match_case(core, best)))
    return " ".join(out)


def collapse_repeats(text: str, max_repeat: int = 3) -> str:
    words = text.split()
    for size in (1, 2, 3, 4):
        i = 0
        while i + size * (max_repeat + 1) <= len(words):
            block = words[i:i + size]
            reps, j = 1, i + size
            while j + size <= len(words) and words[j:j + size] == block:
                reps += 1
                j += size
            if reps > max_repeat:
                words = words[:i + size * max_repeat] + words[j:]
            i += size
    return " ".join(words)


def _align_to_pivot(pivot: list, hyp: list) -> tuple[list, list]:
    align: list = [""] * len(pivot)
    inserts: list = [[] for _ in range(len(pivot) + 1)]
    i = j = 0
    for op in Levenshtein.editops(pivot, hyp):
        while i < op.src_pos:
            align[i] = hyp[j]
            i += 1
            j += 1
        if op.tag == "replace":
            align[i] = hyp[op.dest_pos]
            i += 1
            j += 1
        elif op.tag == "delete":
            align[i] = ""
            i += 1
        elif op.tag == "insert":
            inserts[i].append(hyp[op.dest_pos])
            j += 1
    while i < len(pivot):
        align[i] = hyp[j] if j < len(hyp) else ""
        i += 1
        j += 1
    return align, inserts


def rover_fuse(hyps: Sequence[str], weights: Sequence[float], level: str = "char") -> str:
    pairs = [(h, w) for h, w in zip(hyps, weights) if isinstance(h, str)]
    if not pairs:
        return ""
    merged: dict[str, float] = defaultdict(float)
    for h, w in pairs:
        merged[h] += float(w)
    if len(merged) == 1:
        return next(iter(merged))
    items = sorted(merged.items(), key=lambda kv: -kv[1])
    total = sum(w for _, w in items)
    split = (lambda s: list(s)) if level == "char" else (lambda s: s.split())
    pivot = split(items[0][0])
    if not pivot:
        return items[0][0]
    slot_votes = [defaultdict(float) for _ in range(len(pivot))]
    gap_votes = [defaultdict(float) for _ in range(len(pivot) + 1)]
    for text, w in items:
        align, inserts = _align_to_pivot(pivot, split(text))
        for k, unit in enumerate(align):
            slot_votes[k][unit] += w
        for k, run in enumerate(inserts):
            gap_votes[k][tuple(run)] += w
    out: list = []
    for k in range(len(pivot) + 1):
        run, vote = max(gap_votes[k].items(), key=lambda kv: (kv[1], -len(kv[0])))
        if run and vote > total * 0.5:
            out.extend(run)
        if k < len(pivot):
            best_unit, _ = max(slot_votes[k].items(),
                               key=lambda kv: (kv[1], kv[0] == pivot[k], -len(kv[0])))
            if best_unit:
                out.append(best_unit)
    return "".join(out) if level == "char" else " ".join(out)


assert rover_fuse(["hello world", "hello world", "hallo world"], [1, 1, 1]) == "hello world"
print("[rover] ok")

# %%
@dataclass
class PostConfig:
    w_score: float = 1.0
    w_votes: float = 0.45
    w_lm: float = 0.15
    w_oov: float = -0.30
    w_lenz: float = -0.02
    use_rover: bool = True
    rover_level: str = "char"
    rover_temp: float = 0.08
    repair: bool = True
    repair_sim: float = 90.0
    repair_freq: int = 3
    restrict_charset: bool = True
    collapse: bool = True
    use_vlm: bool = False
    w_vlm: float = 0.35


FOLD_LM: dict[int, CharLM] = {}
FOLD_LEX: dict[int, Lexicon] = {}
FOLD_OF = dict(zip(train_df.ID, train_df.fold))


def fold_resources(fold: int) -> tuple[CharLM, Lexicon]:
    if fold not in FOLD_LM:
        texts = train_df.loc[train_df.fold != fold, "Target"].tolist() if fold >= 0 \
            else train_df["Target"].tolist()
        FOLD_LM[fold] = CharLM(texts, order=CFG.lm_order)
        FOLD_LEX[fold] = Lexicon(texts)
    return FOLD_LM[fold], FOLD_LEX[fold]


def frame_to_nbest(df: pd.DataFrame, family: str) -> dict:
    out: dict[str, list] = defaultdict(list)
    for row in df.itertuples():
        out[row.ID].append({"text": normalize_text(row.text), "score": float(row.score),
                            "view": getattr(row, "view", family), "family": family})
    return dict(out)


def merge_nbest(base: dict, extra: dict) -> dict:
    merged = {k: list(v) for k, v in base.items()}
    for i, cands in extra.items():
        merged.setdefault(i, []).extend(cands)
    return merged


OOF_NBEST: dict = {}
TEST_NBEST: dict = {}
for d in DECODES:
    OOF_NBEST = merge_nbest(OOF_NBEST, frame_to_nbest(d["oof"], d["family"]))
    TEST_NBEST = merge_nbest(TEST_NBEST, frame_to_nbest(d["test"], d["family"]))

OOF_IDS = sorted({i for i in OOF_NBEST})
OOF_REFS = train_df.set_index("ID").loc[OOF_IDS, "Target"].tolist()


def load_vlm_csv(path: str, family: str = "vlm") -> dict:
    p = Path(path)
    if not p.exists():
        p = WORK / path
    if not p.exists():
        return {}
    df = pd.read_csv(p, keep_default_na=False)
    df["ID"] = df["ID"].astype(str).str.strip()
    df["Target"] = df["Target"].map(normalize_text)
    return {r.ID: [{"text": r.Target, "score": 0.0, "view": "vlm", "family": family}]
            for r in df.itertuples()}


VLM_TEST = load_vlm_csv(CFG.vlm_test_csv)
VLM_OOF = load_vlm_csv(CFG.vlm_oof_csv) if CFG.vlm_oof_csv else {}
if VLM_TEST:
    print(f"[vlm] loaded {len(VLM_TEST)} test lines from {CFG.vlm_test_csv}")
    TEST_NBEST = merge_nbest(TEST_NBEST, VLM_TEST)
if VLM_OOF:
    print(f"[vlm] loaded {len(VLM_OOF)} OOF lines from {CFG.vlm_oof_csv}")
    OOF_NBEST = merge_nbest(OOF_NBEST, VLM_OOF)
HAS_VLM_OOF = bool(VLM_OOF) and any(c.get("family") == "vlm" for i in OOF_IDS
                                    for c in OOF_NBEST.get(i, []))


def clean_candidate(text: str, restrict: bool, collapse: bool) -> str:
    text = normalize_text(text)
    if collapse:
        text = collapse_repeats(text)
    if restrict:
        text = normalize_text(restrict_charset(text))
    return text


def build_features(nbest: dict, ids: Sequence[str], fold_of: dict,
                   restrict: bool = True, collapse: bool = True) -> dict:
    feats = {}
    for image_id in progress(ids, desc="features", unit="line", leave=False):
        lm, lex = fold_resources(fold_of.get(image_id, -1))
        agg: dict[str, dict] = defaultdict(lambda: {"scores": [], "votes": 0.0, "vlm": 0.0})
        for c in nbest.get(image_id, []):
            text = clean_candidate(c["text"], restrict, collapse)
            rec = agg[text]
            rec["scores"].append(float(c["score"]))
            rec["votes"] += 1.0
            if c.get("family") == "vlm":
                rec["vlm"] += 1.0
        if not agg:
            agg[""] = {"scores": [-50.0], "votes": 1.0, "vlm": 0.0}
        texts, rows = [], []
        for text, rec in agg.items():
            s = np.asarray(rec["scores"], dtype=np.float64)
            rows.append([float(s.max()), float(s.mean()), math.log1p(rec["votes"]),
                         lm.logprob_per_char(text), lex.oov_rate(text),
                         abs((len(text) - lex.mean_chars) / lex.std_chars),
                         rec["vlm"]])
            texts.append(text)
        feats[image_id] = {"texts": texts, "M": np.asarray(rows, dtype=np.float64)}
    return feats


def combined_scores(M: np.ndarray, pc: PostConfig) -> np.ndarray:
    return (pc.w_score * (0.5 * M[:, 0] + 0.5 * M[:, 1])
            + pc.w_votes * M[:, 2]
            + pc.w_lm * M[:, 3]
            + pc.w_oov * M[:, 4]
            + pc.w_lenz * M[:, 5]
            + (pc.w_vlm * M[:, 6] if pc.use_vlm else 0.0))


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
            pred = (rover_fuse([t for t, k in zip(texts, keep) if k],
                               w[keep].tolist(), level=pc.rover_level)
                    if keep.sum() >= 2 else texts[int(np.argmax(s))])
        else:
            pred = texts[int(np.argmax(s))]
        if not pred.strip():
            for j in np.argsort(-s):
                if texts[j].strip():
                    pred = texts[j]
                    break
        if pc.repair and lexicon_for is not None:
            pred = repair_words(pred, lexicon_for(image_id),
                                min_sim=pc.repair_sim, min_freq=pc.repair_freq)
        out[image_id] = normalize_text(pred)
    return out


stage("post-processing weight search")
OOF_FEATS = build_features(OOF_NBEST, OOF_IDS, FOLD_OF)


def oof_score(pc: PostConfig) -> float:
    preds = apply_post(OOF_FEATS, OOF_IDS, pc,
                       lexicon_for=lambda i: fold_resources(FOLD_OF.get(i, -1))[1])
    return competition_score(OOF_REFS, [preds[i] for i in OOF_IDS])["score"]


base_pc = PostConfig(use_rover=False, repair=False, use_vlm=False)
best_pc, best_val = base_pc, oof_score(base_pc)
print(f"  identity-only / default weights  {best_val:.5f}")
rng = np.random.default_rng(0)
bar = progress(range(CFG.lm_tune_iters), desc="tune", unit="trial")
for it in bar:
    trial = PostConfig(
        w_score=float(rng.uniform(0.0, 2.0)),
        w_votes=float(rng.uniform(0.0, 1.5)),
        w_lm=float(rng.uniform(0.0, 1.2)),
        w_oov=float(rng.uniform(-1.5, 0.1)),
        w_lenz=float(rng.uniform(-0.3, 0.05)),
        use_rover=False, repair=False, use_vlm=HAS_VLM_OOF,
        w_vlm=float(rng.uniform(0.0, 1.0)) if HAS_VLM_OOF else 0.0,
    )
    val = oof_score(trial)
    if val > best_val:
        best_pc, best_val = trial, val
        bar.write(f"  it {it:4d}  {val:.5f}")
    bar.set_postfix_str(f"best {best_val:.5f}", refresh=False)
bar.close()

stage_b = [(best_pc, best_val, "rover off")]
for level in ("char", "word"):
    for temp in (0.03, 0.08, 0.2):
        trial = PostConfig(**{**asdict(best_pc), "use_rover": True,
                              "rover_level": level, "rover_temp": temp})
        val = oof_score(trial)
        stage_b.append((trial, val, f"{level} T={temp}"))
        print(f"  rover {level} T={temp} {val:.5f}")
best_pc, best_val, chosen = max(stage_b, key=lambda kv: kv[1])
print(f"[tune] rover {chosen} -> {best_val:.5f}")

stage_c = [(best_pc, best_val, "repair off")]
for sim in (86.0, 90.0, 94.0):
    for freq in (2, 5, 15):
        trial = PostConfig(**{**asdict(best_pc), "repair": True,
                              "repair_sim": sim, "repair_freq": freq})
        val = oof_score(trial)
        stage_c.append((trial, val, f"sim={sim} freq={freq}"))
best_pc, best_val, chosen = max(stage_c, key=lambda kv: kv[1])
print(f"[tune] repair {chosen} -> {best_val:.5f}")
POST = best_pc
print(json.dumps(asdict(POST), indent=2))
