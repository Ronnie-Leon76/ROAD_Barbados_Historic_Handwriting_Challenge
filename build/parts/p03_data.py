# %% [markdown]
# ## 3. Data
#
# `images.zip` (≈421 MB, 5 472 crops) is fetched from the URL on the challenge Data
# tab and unpacked once. The CSVs are expected next to the notebook. Every ID is
# resolved through a single filename index so a nested or renamed extraction
# directory does not break anything.

# %%
DATA = WORK / CFG.data_dir
DATA.mkdir(parents=True, exist_ok=True)


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[data] {dest.name} already present ({dest.stat().st_size/1e6:.0f} MB)")
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
    # The archive may unpack to data/images or data/<something>/images
    candidates = [p for p in [img_dir, DATA] if p.exists()]
    for root in candidates:
        jpgs = list(root.rglob("*.jpg"))
        if jpgs:
            return jpgs[0].parent
    raise FileNotFoundError("No .jpg files found after extraction")


stage("data: download and index")
IMAGE_DIR = ensure_images()
IMAGE_INDEX = {p.stem: p for p in IMAGE_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}}
print(f"[data] image dir  {IMAGE_DIR}")
print(f"[data] indexed    {len(IMAGE_INDEX)} images")


def image_path(image_id: str) -> Path | None:
    return IMAGE_INDEX.get(str(image_id).strip())


train_df = pd.read_csv(WORK / CFG.train_csv)
test_df = pd.read_csv(WORK / CFG.test_csv)
sample_sub = pd.read_csv(WORK / CFG.sample_sub_csv)

train_df["ID"] = train_df["ID"].astype(str).str.strip()
test_df["ID"] = test_df["ID"].astype(str).str.strip()

missing_train = [i for i in train_df.ID if image_path(i) is None]
missing_test = [i for i in test_df.ID if image_path(i) is None]
print(f"[data] train rows {len(train_df)} | missing images {len(missing_train)}")
print(f"[data] test rows  {len(test_df)} | missing images {len(missing_test)}")
assert not missing_test, f"Test images missing: {missing_test[:5]}"
if missing_train:
    train_df = train_df[~train_df.ID.isin(missing_train)].reset_index(drop=True)

# %% [markdown]
# ### 3.1 Label handling
#
# The transcriptions use archival conventions that must survive round-tripping:
# `^` marks a superscript (`y^e`, `15^th`, `w^tsoever`), `~` marks a scribal
# abbreviation stroke, and original spelling is preserved (`publique`, `prsence`).
# Normalisation is therefore deliberately minimal — Unicode look-alikes are folded
# onto the ASCII characters that actually occur in the training labels, whitespace is
# collapsed, and nothing else is touched.

# %%
import unicodedata

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
    """Minimal, loss-free-for-this-corpus label normalisation."""
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
print(f"[labels] {len(train_df)} usable rows | {len(CHARSET)} distinct characters")
print("[labels] charset:", "".join(CHARSET))

# Characters seen fewer than 3 times are almost certainly annotation noise; a
# prediction containing them is more likely a hallucination than a real glyph.
RARE_CHARS = {c for c, n in CHAR_FREQ.items() if n < 3}
print("[labels] rare chars (<3 occurrences):", sorted(RARE_CHARS))

WORD_FREQ = Counter(w for t in train_df["Target"] for w in t.split())
VOCAB = set(WORD_FREQ)
print(f"[labels] vocabulary {len(VOCAB)} types / {sum(WORD_FREQ.values())} tokens")
print("[labels] most common:", WORD_FREQ.most_common(15))


def restrict_charset(s: str) -> str:
    """Drop characters the model can only have hallucinated."""
    return "".join(ch for ch in s if ch in ALLOWED)

# %% [markdown]
# ### 3.2 Exploratory analysis
#
# Two facts from this cell drive the whole image pipeline:
#
# * **Aspect ratios are extreme** (median ≈ 16.6:1, p90 ≈ 23.7:1). A naive square
#   resize destroys the glyphs, so `Section 6` normalises height and lets width run.
# * **Crops include slivers of the lines above and below.** The label covers only the
#   centre line, so vertical ink-cropping is disabled and the prompt tells the model
#   explicitly to transcribe the central line only.

# %%
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

def image_stats(ids: Sequence[str], limit: int = 1500) -> pd.DataFrame:
    rows = []
    for i in list(ids)[:limit]:
        p = image_path(i)
        if p is None:
            continue
        with Image.open(p) as im:
            w, h = im.size
        rows.append({"ID": i, "w": w, "h": h, "aspect": w / max(h, 1)})
    return pd.DataFrame(rows)


tr_stats = image_stats(train_df.ID)
te_stats = image_stats(test_df.ID)
tr_stats = tr_stats.merge(train_df, on="ID", how="left")
tr_stats["n_chars"] = tr_stats["Target"].str.len()
tr_stats["n_words"] = tr_stats["Target"].str.split().apply(len)
tr_stats["px_per_char"] = tr_stats["w"] / tr_stats["n_chars"].clip(lower=1)

summary = pd.DataFrame({
    "train": tr_stats[["w", "h", "aspect", "n_chars", "n_words", "px_per_char"]].median(),
    "train_p90": tr_stats[["w", "h", "aspect", "n_chars", "n_words", "px_per_char"]].quantile(0.9),
})
print(summary.round(2))
print("\ntest image geometry vs train (median / p90):")
print(pd.DataFrame({
    "test_med": te_stats[["w", "h", "aspect"]].median(),
    "test_p90": te_stats[["w", "h", "aspect"]].quantile(0.9),
    "train_med": tr_stats[["w", "h", "aspect"]].median(),
    "train_p90": tr_stats[["w", "h", "aspect"]].quantile(0.9),
}).round(1))

fig, ax = plt.subplots(2, 3, figsize=(16, 7))
ax[0, 0].hist(tr_stats.aspect, bins=50, color="#3b6ea5")
ax[0, 0].hist(te_stats.aspect, bins=50, color="#c8553d", alpha=0.55)
ax[0, 0].set_title("aspect ratio (blue=train, red=test)")
ax[0, 1].hist(tr_stats.h, bins=50, color="#3b6ea5")
ax[0, 1].set_title("crop height (px)")
ax[0, 2].hist(tr_stats.n_chars, bins=40, color="#3b6ea5")
ax[0, 2].set_title("label length (chars)")
ax[1, 0].hist(tr_stats.n_words, bins=range(0, 26), color="#3b6ea5")
ax[1, 0].set_title("label length (words)")
ax[1, 1].hist(np.log10(tr_stats.px_per_char), bins=40, color="#3b6ea5")
ax[1, 1].set_title("log10 px per character")
top = CHAR_FREQ.most_common(30)
ax[1, 2].bar([c for c, _ in top], [n for _, n in top], color="#3b6ea5")
ax[1, 2].set_yscale("log")
ax[1, 2].set_title("character frequency (top 30)")
plt.tight_layout()
plt.savefig(OUT / "eda_overview.png", dpi=110)
plt.close(fig)
print(f"\n[eda] saved {OUT/'eda_overview.png'}")

# %%
# Visual check: what the model actually sees, and why vertical cropping is off.
fig, axes = plt.subplots(6, 1, figsize=(16, 9))
for ax_, (_, row) in zip(axes, train_df.sample(6, random_state=3).iterrows()):
    ax_.imshow(Image.open(image_path(row.ID)).convert("L"), cmap="gray")
    ax_.set_title(row.Target, fontsize=8, loc="left")
    ax_.axis("off")
plt.tight_layout()
plt.savefig(OUT / "eda_samples.png", dpi=110)
plt.close(fig)
print(f"[eda] saved {OUT/'eda_samples.png'} - note the partial lines at top/bottom")
