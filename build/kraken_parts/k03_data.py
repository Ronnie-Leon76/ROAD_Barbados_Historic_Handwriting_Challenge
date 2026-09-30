# %% [markdown]
# ## 3. Data, labels, metric, split

# %%
DATA = WORK / CFG.data_dir
DATA.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[data] {dest.name} already present ({dest.stat().st_size / 1e6:.0f} MB)")
        return dest
    import urllib.request
    print(f"[data] downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as fh:
        total = int(resp.headers.get("Content-Length", 0)) or None
        bar = progress(total=total, desc="download", unit="B",
                       unit_scale=True, unit_divisor=1024, bar_format=None)
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            fh.write(buf)
            bar.update(len(buf))
        bar.close()
    tmp.rename(dest)
    return dest


def ensure_images() -> Path:
    zip_path = DATA / "images.zip"
    img_dir = DATA / CFG.image_subdir
    if not (img_dir.exists() and any(img_dir.rglob("*.jpg"))):
        download(CFG.images_url, zip_path)
        print("[data] extracting ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA)
    for root in [img_dir, DATA]:
        if root.exists():
            jpgs = list(root.rglob("*.jpg"))
            if jpgs:
                return jpgs[0].parent
    raise FileNotFoundError("No .jpg files found after extraction")


stage("data: download and index")
IMAGE_DIR = ensure_images()
IMAGE_INDEX = {p.stem: p for p in IMAGE_DIR.rglob("*")
               if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}}
print(f"[data] image dir  {IMAGE_DIR} | indexed {len(IMAGE_INDEX)}")


def image_path(image_id: str) -> Path | None:
    return IMAGE_INDEX.get(str(image_id).strip())


train_df = pd.read_csv(WORK / CFG.train_csv)
test_df = pd.read_csv(WORK / CFG.test_csv)
sample_sub = pd.read_csv(WORK / CFG.sample_sub_csv)
train_df["ID"] = train_df["ID"].astype(str).str.strip()
test_df["ID"] = test_df["ID"].astype(str).str.strip()

UNICODE_FOLD = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00a0": " ", "\u2007": " ", "\u202f": " ", "\u2009": " ", "\u200b": "",
    "\u2026": "...", "\ufeff": "",
}
_FOLD_RE = re.compile("|".join(map(re.escape, UNICODE_FOLD)))


def normalize_text(s: Any) -> str:
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return ""
    s = unicodedata.normalize("NFC", str(s))
    s = _FOLD_RE.sub(lambda m: UNICODE_FOLD[m.group()], s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return " ".join(s.split()).strip()


train_df["Target"] = train_df["Target"].map(normalize_text)
train_df = train_df[train_df["Target"].str.len() > 0].reset_index(drop=True)
CHARSET = sorted(Counter("".join(train_df["Target"])).keys())
CHAR_FREQ = Counter("".join(train_df["Target"]))
ALLOWED = set(CHARSET)
print(f"[labels] {len(train_df)} rows | {len(CHARSET)} chars | {''.join(CHARSET)!r}")


def restrict_charset(s: str) -> str:
    return "".join(ch for ch in s if ch in ALLOWED)

# %%
from rapidfuzz.distance import Levenshtein

WEIGHT_FACTOR = 0.5


def _tokens(s: str, level: str) -> list:
    return list(s) if level == "char" else s.split()


def error_rates(refs: Sequence[str], hyps: Sequence[str], level: str,
                weight_factor: float = WEIGHT_FACTOR) -> dict:
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
    return {"power": num_p / max(den_p, 1e-9), "micro": num_m / max(den_m, 1e-9),
            "macro": float(np.mean(per_sample)) if per_sample else 1.0,
            "per_sample": np.asarray(per_sample)}


def competition_score(refs: Sequence[str], hyps: Sequence[str],
                      weight_factor: float = WEIGHT_FACTOR, verbose: bool = False) -> dict:
    cer = error_rates(refs, hyps, "char", weight_factor)
    wer = error_rates(refs, hyps, "word", weight_factor)
    out = {"score": 0.5 * (1 - wer["power"]) + 0.5 * (1 - cer["power"]),
           "cer": cer["power"], "wer": wer["power"], "n": len(refs)}
    if verbose:
        print(f"score {out['score']:.5f} | CER {out['cer']:.4f} | WER {out['wer']:.4f} | n={out['n']}")
    return out


def _test_metric() -> None:
    assert abs(competition_score(["the quick brown fox"], ["the quick brown fox"])["score"] - 1) < 1e-9
    assert abs(competition_score(["abc def"], [""])["score"]) < 1e-9
    one = competition_score(["abcd"], ["abcx"])
    assert abs(one["cer"] - 0.25) < 1e-9 and abs(one["wer"] - 1.0) < 1e-9
    print("[metric] unit tests passed")


_test_metric()

# %%
from rapidfuzz import process as rf_process
from sklearn.model_selection import StratifiedGroupKFold


def build_text_groups(texts: Sequence[str], threshold: float = 0.90) -> np.ndarray:
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

    by_key: dict[str, list[int]] = defaultdict(list)
    for i, k in enumerate(keys):
        by_key[k].append(i)
    for idxs in by_key.values():
        for j in idxs[1:]:
            union(idxs[0], j)
    sim = rf_process.cdist(keys, keys, scorer=Levenshtein.normalized_similarity,
                           score_cutoff=threshold, workers=-1, dtype=np.float32)
    rows, cols = np.nonzero(sim)
    for i, j in zip(rows, cols):
        if i < j:
            union(int(i), int(j))
    roots = np.array([find(i) for i in range(n)])
    _, groups = np.unique(roots, return_inverse=True)
    return groups


train_df["group"] = build_text_groups(train_df["Target"].tolist())
train_df["n_chars"] = train_df["Target"].str.len()
train_df["n_words"] = train_df["Target"].str.split().apply(len)
train_df["len_bin"] = pd.qcut(train_df["n_chars"], q=6, labels=False, duplicates="drop")
sgkf = StratifiedGroupKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.fold_seed)
train_df["fold"] = -1
for fold, (_, val_idx) in enumerate(sgkf.split(train_df, train_df["len_bin"], train_df["group"])):
    train_df.loc[train_df.index[val_idx], "fold"] = fold
leaks = train_df.groupby("group")["fold"].nunique()
assert (leaks == 1).all(), "groups leak across folds"
print(train_df.groupby("fold").agg(n=("ID", "size"), med_chars=("n_chars", "median")))
print("[split] no group leaks")
train_df.to_csv(OUT / "train_folds.csv", index=False)


def fold_frames(fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (train_df[train_df.fold != fold].reset_index(drop=True),
            train_df[train_df.fold == fold].reset_index(drop=True))
