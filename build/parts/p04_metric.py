# %% [markdown]
# ## 4. The competition metric
#
# From the Evaluation tab: two error metrics, **weighted WER** and **weighted CER**,
# combined as `0.5 * WER_score + 0.5 * CER_score` where each score is `1 - error`.
# "Longer reference transcriptions are weighted more heavily, so errors on longer
# samples contribute more to the final score than errors on very short samples."
#
# That description matches a length-power weighting, `w_i = len(ref_i) ** f`, with
# the starter pack's default `weight_factor = 0.5`. Because the exact server-side
# formula is not published, three variants are computed:
#
# * `power` — per-sample error rate averaged with weights `len ** f` (**primary**),
# * `micro` — pooled edits over pooled reference length (`f = 1`, the classic
#   corpus-level WER/CER),
# * `macro` — unweighted mean of per-sample error rates.
#
# All three move together; `power` is used for model selection and the others are
# printed as a guard against over-fitting one particular definition. Missing or empty
# predictions are scored as fully incorrect, exactly as the rules state.

# %%
from rapidfuzz.distance import Levenshtein

WEIGHT_FACTOR = 0.5


def _tokens(s: str, level: str) -> list:
    return list(s) if level == "char" else s.split()


def error_rates(refs: Sequence[str], hyps: Sequence[str], level: str,
                weight_factor: float = WEIGHT_FACTOR) -> dict:
    """Return power / micro / macro error rates for one level (char or word)."""
    num_p = den_p = num_m = den_m = 0.0
    per_sample = []
    for ref, hyp in zip(refs, hyps):
        r = _tokens(ref, level)
        h = _tokens(hyp if isinstance(hyp, str) else "", level)
        n = max(len(r), 1)
        d = Levenshtein.distance(r, h)
        rate = d / n
        per_sample.append(rate)
        w = n ** weight_factor
        num_p += w * rate
        den_p += w
        num_m += d
        den_m += n
    return {
        "power": num_p / max(den_p, 1e-9),
        "micro": num_m / max(den_m, 1e-9),
        "macro": float(np.mean(per_sample)) if per_sample else 1.0,
        "per_sample": np.asarray(per_sample),
    }


def competition_score(refs: Sequence[str], hyps: Sequence[str],
                      weight_factor: float = WEIGHT_FACTOR,
                      verbose: bool = False) -> dict:
    """Leaderboard-style score: 0.5*(1-WER) + 0.5*(1-CER)."""
    cer = error_rates(refs, hyps, "char", weight_factor)
    wer = error_rates(refs, hyps, "word", weight_factor)
    out = {
        "score": 0.5 * (1 - wer["power"]) + 0.5 * (1 - cer["power"]),
        "cer": cer["power"],
        "wer": wer["power"],
        "cer_micro": cer["micro"],
        "wer_micro": wer["micro"],
        "score_micro": 0.5 * (1 - wer["micro"]) + 0.5 * (1 - cer["micro"]),
        "cer_macro": cer["macro"],
        "wer_macro": wer["macro"],
        "n": len(refs),
    }
    if verbose:
        print(f"score {out['score']:.5f} | CER {out['cer']:.4f} | WER {out['wer']:.4f} "
              f"| (micro: score {out['score_micro']:.5f}, CER {out['cer_micro']:.4f}, "
              f"WER {out['wer_micro']:.4f}) | n={out['n']}")
    return out


def score_frames(gt: pd.DataFrame, pred: pd.DataFrame, id_col: str = "ID",
                 text_col: str = "Target", verbose: bool = True) -> dict:
    """Join on ID the way the grader does: unmatched IDs count as empty predictions."""
    p = pred.set_index(id_col)[text_col].map(normalize_text).to_dict()
    refs = [normalize_text(t) for t in gt[text_col]]
    hyps = [p.get(i, "") for i in gt[id_col]]
    missing = sum(1 for i in gt[id_col] if i not in p)
    if missing:
        print(f"!! {missing} reference IDs have no prediction - scored as empty")
    return competition_score(refs, hyps, verbose=verbose)


# --------------------------------------------------------------------------
# Unit tests: a metric you have not tested is a metric you do not have.
# --------------------------------------------------------------------------
def _test_metric() -> None:
    perfect = competition_score(["the quick brown fox"], ["the quick brown fox"])
    assert abs(perfect["score"] - 1.0) < 1e-9, perfect

    empty = competition_score(["abc def"], [""])
    assert abs(empty["cer"] - 1.0) < 1e-9 and abs(empty["wer"] - 1.0) < 1e-9, empty
    assert abs(empty["score"]) < 1e-9, empty

    one_char = competition_score(["abcd"], ["abcx"])
    assert abs(one_char["cer"] - 0.25) < 1e-9, one_char
    assert abs(one_char["wer"] - 1.0) < 1e-9, one_char

    # Longer references must dominate: the same relative error on a long line moves
    # the weighted score more than on a short one.
    long_ref, short_ref = "a" * 100, "a" * 4
    hit_long = competition_score([long_ref, short_ref], ["b" * 100, short_ref])
    hit_short = competition_score([long_ref, short_ref], [long_ref, "b" * 4])
    assert hit_long["cer"] > hit_short["cer"], (hit_long["cer"], hit_short["cer"])

    # Word-level insertion
    ins = error_rates(["a b c"], ["a b c d"], "word")
    assert abs(ins["power"] - 1 / 3) < 1e-9, ins

    # Frame joining with a missing ID
    gt = pd.DataFrame({"ID": ["x", "y"], "Target": ["hello world", "foo bar"]})
    pr = pd.DataFrame({"ID": ["x"], "Target": ["hello world"]})
    joined = score_frames(gt, pr, verbose=False)
    assert 0.4 < joined["score"] < 0.6, joined
    print("[metric] all unit tests passed")


_test_metric()

# Reference points on this dataset, so a validation number can be sanity-checked:
_probe_refs = train_df["Target"].tolist()[:500]
print("[metric] blank submission        ->", round(competition_score(_probe_refs, [""] * 500)["score"], 4))
print("[metric] lowercased references   ->", round(competition_score(_probe_refs, [t.lower() for t in _probe_refs])["score"], 4))
print("[metric] 1 char in 20 corrupted  ->", round(competition_score(
    _probe_refs,
    ["".join("*" if (i % 20 == 7) else c for i, c in enumerate(t)) for t in _probe_refs],
)["score"], 4))
