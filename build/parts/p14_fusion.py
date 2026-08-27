# %% [markdown]
# ## 15. Fusion, rescoring and repair
#
# Beam search returns the most likely *token* sequence. The leaderboard rewards the
# lowest *edit distance*, and those are not the same objective. Three mechanisms
# close the gap, all tuned on out-of-fold predictions and all reversible if they do
# not help:
#
# 1. **ROVER** — build a confusion network from every hypothesis (all beams, all TTA
#    views, all folds) and take a weighted vote per aligned slot. Voting fixes the
#    single-character disagreements that dominate CER, and can produce a string that
#    no individual hypothesis proposed.
# 2. **N-best rescoring** — re-rank hypotheses with a linear model over beam score,
#    cross-view agreement, a character n-gram language model, out-of-vocabulary rate
#    and length prior. Weights come from random search against the real metric.
# 3. **Lexicon repair** — map out-of-vocabulary words onto near neighbours in the
#    training vocabulary, but only where the LM agrees and the edit distance is small.
#
# **Leakage control:** the language model and the lexicon are rebuilt *per fold* from
# the training portion only. Tuning against an LM that has memorised the validation
# sentences would produce weights that collapse on the leaderboard.

# %%
class CharLM:
    """Character n-gram model with stupid back-off - small, fast, no dependencies."""

    def __init__(self, texts: Sequence[str], order: int = 7, alpha: float = 0.4):
        self.order = order
        self.alpha = alpha
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
        total = 0.0
        for i in range(self.order - 1, len(s)):
            total += self._logp(s[i - self.order + 1:i], s[i])
        return total

    def logprob_per_char(self, text: str) -> float:
        return self.logprob(text) / max(len(text) + 1, 1)


class Lexicon:
    def __init__(self, texts: Sequence[str]):
        self.freq = Counter(w for t in texts for w in t.split())
        self.vocab = set(self.freq)
        self.lower: dict[str, list[str]] = defaultdict(list)
        for w in self.vocab:
            self.lower[w.lower()].append(w)
        self.choices = sorted(self.vocab)   # sorted => deterministic tie-breaking
        self.mean_chars = float(np.mean([len(t) for t in texts])) if len(texts) else 60.0
        self.std_chars = float(np.std([len(t) for t in texts])) or 1.0

    def oov_rate(self, text: str) -> float:
        words = text.split()
        if not words:
            return 1.0
        return sum(w not in self.vocab for w in words) / len(words)


def match_case(source: str, target: str) -> str:
    """Give `target` the capitalisation pattern of `source`."""
    if source.isupper() and len(source) > 1:
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def repair_words(text: str, lex: Lexicon, min_sim: float = 88.0,
                 min_freq: int = 3, max_len_ratio: float = 0.34) -> str:
    """Snap out-of-vocabulary words onto close, frequent training-vocabulary words."""
    out = []
    for word in text.split():
        core = word.strip(".,;:()\"'")
        if not core or core in lex.vocab or len(core) < 3:
            out.append(word)
            continue
        exact_case = lex.lower.get(core.lower())
        if exact_case:  # only the capitalisation differs - keep what the model wrote
            out.append(word)
            continue
        budget = max(1, int(len(core) * max_len_ratio))
        # Look at several neighbours, not just the top one: near-ties are common
        # ("presence" vs "Presence") and only the frequent variant is a safe target.
        cands = rf_process.extract(
            core, lex.choices, scorer=Levenshtein.normalized_similarity,
            processor=str.lower, score_cutoff=min_sim / 100.0, limit=8,
        )
        viable = [
            (sim + 0.02 * math.log10(lex.freq[c] + 1), c)
            for c, sim, _ in cands
            if lex.freq[c] >= min_freq
            and Levenshtein.distance(core.lower(), c.lower()) <= budget
        ]
        if not viable:
            out.append(word)
            continue
        best = max(viable)[1]
        out.append(word.replace(core, match_case(core, best)))
    return " ".join(out)


def collapse_repeats(text: str, max_repeat: int = 3) -> str:
    """Truncate degenerate loops such as 'of the of the of the ...'."""
    words = text.split()
    for size in (1, 2, 3, 4):
        i = 0
        while i + size * (max_repeat + 1) <= len(words):
            block = words[i:i + size]
            reps = 1
            j = i + size
            while j + size <= len(words) and words[j:j + size] == block:
                reps += 1
                j += size
            if reps > max_repeat:
                words = words[:i + size * max_repeat] + words[j:]
            i += size
    return " ".join(words)

# %% [markdown]
# ### 15.1 ROVER
#
# Hypotheses are aligned to the highest-weighted one with Levenshtein edit
# operations, giving a confusion network: one voting slot per pivot unit (a
# substitution or a deletion) plus one slot per gap (an insertion). Each slot is
# decided by weighted majority, so the fused output can contain a unit that only a
# minority of hypotheses proposed at that position while keeping the rest.

# %%
def _align_to_pivot(pivot: list, hyp: list) -> tuple[list, list]:
    """Return (aligned units per pivot slot, inserted runs per gap)."""
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
    """Weighted confusion-network vote over aligned hypotheses."""
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
            best_unit, best_vote = max(
                slot_votes[k].items(),
                key=lambda kv: (kv[1], kv[0] == pivot[k], -len(kv[0])),
            )
            _ = best_vote
            if best_unit:
                out.append(best_unit)
    return "".join(out) if level == "char" else " ".join(out)


def _test_rover() -> None:
    # Majority repairs a single-character error.
    assert rover_fuse(["hello world", "hello world", "hallo world"], [1, 1, 1]) == "hello world"
    # Majority repairs a missing character.
    assert rover_fuse(["presence", "presence", "presnce"], [1, 1, 1]) == "presence"
    # Majority removes a spurious insertion.
    assert rover_fuse(["the deed", "the deed", "the deedx"], [1, 1, 1]) == "the deed"
    # Weights dominate raw counts.
    assert rover_fuse(["abc", "abd", "abd"], [10.0, 1.0, 1.0]) == "abc"
    # Word level works too.
    assert rover_fuse(["a b c", "a b c", "a x c"], [1, 1, 1], level="word") == "a b c"
    # Degenerate inputs.
    assert rover_fuse([], []) == ""
    assert rover_fuse(["only"], [1.0]) == "only"
    # Fusion can beat every input (different errors in different places).
    fused = rover_fuse(["the quick brown fex", "the quick brawn fox", "the quick brown fox"], [1, 1, 1])
    assert fused == "the quick brown fox", fused
    print("[rover] all unit tests passed")


_test_rover()
