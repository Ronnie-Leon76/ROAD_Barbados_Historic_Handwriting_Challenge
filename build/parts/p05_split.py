# %% [markdown]
# ## 5. Cross-validation split
#
# The corpus contains repeated legal formulae ("Sealed and delivered in the presence
# of", "To all Christian People to whome these presents shall come") captured from
# different scans. If a near-duplicate line lands in both train and validation the
# local score becomes optimistic and checkpoint selection degrades.
#
# Lines are therefore clustered by text similarity (union-find over pairs with
# normalised Levenshtein similarity ≥ 0.90) and the clusters are kept intact by
# `StratifiedGroupKFold`, stratified on label length so every fold sees the same mix
# of short and long lines.

# %%
from rapidfuzz import process as rf_process
from sklearn.model_selection import StratifiedGroupKFold


def build_text_groups(texts: Sequence[str], threshold: float = 0.90) -> np.ndarray:
    """Union-find clustering of near-duplicate transcriptions."""
    keys = [" ".join(re.sub(r"[^a-z0-9 ]", " ", t.lower()).split()) for t in texts]
    n = len(keys)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    # Exact matches first (cheap), then fuzzy neighbours.
    by_key: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_key[k].append(i)
    for idxs in by_key.values():
        for j in idxs[1:]:
            union(idxs[0], j)

    sim = rf_process.cdist(
        keys, keys, scorer=Levenshtein.normalized_similarity,
        score_cutoff=threshold, workers=-1, dtype=np.float32,
    )
    rows, cols = np.nonzero(sim)
    for i, j in zip(rows, cols):
        if i < j:
            union(int(i), int(j))

    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


t0 = time.time()
train_df["group"] = build_text_groups(train_df["Target"].tolist())
n_groups = train_df["group"].nunique()
clustered = int((train_df.groupby("group").size() > 1).sum())
print(f"[split] {n_groups} text groups from {len(train_df)} lines "
      f"({clustered} groups hold near-duplicates) in {time.time()-t0:.1f}s")

train_df["n_chars"] = train_df["Target"].str.len()
train_df["n_words"] = train_df["Target"].str.split().apply(len)
train_df["len_bin"] = pd.qcut(train_df["n_chars"], q=6, labels=False, duplicates="drop")

sgkf = StratifiedGroupKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.fold_seed)
train_df["fold"] = -1
for fold, (_, val_idx) in enumerate(sgkf.split(train_df, train_df["len_bin"], train_df["group"])):
    train_df.loc[train_df.index[val_idx], "fold"] = fold

assert (train_df["fold"] >= 0).all()
print(train_df.groupby("fold").agg(
    n=("ID", "size"), med_chars=("n_chars", "median"), med_words=("n_words", "median")
))

# Leakage check: no group may appear in two folds.
leaks = train_df.groupby("group")["fold"].nunique()
assert (leaks == 1).all(), f"{int((leaks > 1).sum())} groups leak across folds"
print("[split] no group leaks across folds")

train_df.to_csv(OUT / "train_folds.csv", index=False)


def fold_frames(fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df[train_df.fold != fold].reset_index(drop=True)
    va = train_df[train_df.fold == fold].reset_index(drop=True)
    return tr, va
