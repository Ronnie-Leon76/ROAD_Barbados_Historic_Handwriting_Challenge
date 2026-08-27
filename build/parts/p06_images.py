# %% [markdown]
# ## 6. Image pipeline
#
# Line crops are 6:1 to 34:1 and the token budget of a ViT is spent per 32×32 pixel
# cell, so the geometry decisions here matter more than anything else in the model.
#
# 1. **Background flattening** — divide the crop by a heavily median-blurred copy of
#    itself. This removes parchment staining and uneven scanner illumination while
#    leaving ink contrast intact.
# 2. **Deskew** — the rotation in ±3° that maximises the variance of the horizontal
#    ink projection, estimated on a 64px-tall thumbnail so it costs ~1 ms.
# 3. **Horizontal ink crop only** — blank left/right margins are trimmed. Vertical
#    cropping is *not* performed: the crops deliberately contain slivers of the lines
#    above and below, and cutting them away would sometimes remove ascenders and
#    descenders of the target line itself.
# 4. **Height normalisation to the patch grid** — resize to `line_height`, cap width
#    at `max_line_width`, and snap both sides to a multiple of 32 so the processor's
#    own `smart_resize` becomes a no-op and the token count is predictable.
#
# At the defaults a median line becomes 160×2656 → 5×83 = 415 visual tokens, roughly
# 3 tokens per character.

# %%
import cv2

cv2.setNumThreads(0)  # dataloader workers already provide the parallelism

CACHE_DIR = OUT / "cache" / f"h{CFG.line_height}_w{CFG.max_line_width}"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def read_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:  # non-ASCII paths on Windows
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def flatten_background(gray: np.ndarray) -> np.ndarray:
    k = max(11, (min(gray.shape[:2]) // 2) | 1)
    k = min(k, 51) | 1
    bg = cv2.medianBlur(gray, k)
    out = cv2.divide(gray, np.maximum(bg, 1), scale=255)
    return np.clip(out, 0, 255).astype(np.uint8)


def ink_mask(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    m = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 41, 12)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def estimate_skew(gray: np.ndarray, max_deg: float = 3.0, step: float = 0.5) -> float:
    """Angle maximising the variance of the horizontal ink projection."""
    h, w = gray.shape[:2]
    scale = 64.0 / max(h, 1)
    small = cv2.resize(gray, (max(int(w * scale), 32), 64), interpolation=cv2.INTER_AREA)
    mask = (ink_mask(small) > 0).astype(np.float32)
    if mask.sum() < 20:
        return 0.0
    best_angle, best_score = 0.0, -1.0
    centre = (mask.shape[1] / 2, mask.shape[0] / 2)
    for angle in np.arange(-max_deg, max_deg + 1e-6, step):
        if abs(angle) < 1e-6:
            rot = mask
        else:
            M = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
            rot = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]),
                                 flags=cv2.INTER_NEAREST, borderValue=0)
        score = float(np.var(rot.sum(axis=1)))
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return best_angle


def rotate_keep(gray: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-3:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def horizontal_ink_crop(gray: np.ndarray, pad_frac: float = 0.02) -> np.ndarray:
    m = ink_mask(gray)
    col = m.sum(axis=0)
    thr = max(col.max() * 0.02, 1.0)
    on = np.nonzero(col > thr)[0]
    if on.size < 2:
        return gray
    pad = int(gray.shape[0] * pad_frac) + 2
    a = max(int(on[0]) - pad, 0)
    b = min(int(on[-1]) + pad, gray.shape[1])
    if b - a < gray.shape[0]:  # never crop into a sliver
        return gray
    return gray[:, a:b]


def snap(x: float, grid: int, lo: int, hi: int) -> int:
    v = int(round(x / grid)) * grid
    return int(min(max(v, max(lo, grid)), hi))


def target_size(w: int, h: int, cfg: Config, scale: float = 1.0) -> tuple[int, int]:
    """Aspect-preserving size on the patch grid."""
    s = (cfg.line_height * scale) / max(h, 1)
    s = min(s, cfg.max_upscale)
    tw, th = w * s, h * s
    if tw > cfg.max_line_width:
        k = cfg.max_line_width / tw
        tw, th = tw * k, th * k
    th = snap(th, cfg.patch_grid, cfg.min_line_height, cfg.line_height * 2)
    tw = snap(tw, cfg.patch_grid, cfg.min_line_width, cfg.max_line_width)
    return int(tw), int(th)


def normalize_line(gray: np.ndarray, cfg: Config, scale: float = 1.0) -> np.ndarray:
    if cfg.background_flatten:
        gray = flatten_background(gray)
    if cfg.deskew:
        gray = rotate_keep(gray, estimate_skew(gray, cfg.deskew_max_deg))
    if cfg.ink_crop_horizontal:
        gray = horizontal_ink_crop(gray)
    if cfg.clahe:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    h, w = gray.shape[:2]
    tw, th = target_size(w, h, cfg, scale)
    interp = cv2.INTER_AREA if (tw < w) else cv2.INTER_CUBIC
    return cv2.resize(gray, (tw, th), interpolation=interp)


def load_line(image_id: str, cfg: Config = None, scale: float = 1.0,
              use_cache: bool = True) -> np.ndarray:
    """Normalised grayscale line image, cached at scale 1.0."""
    cfg = cfg or CFG
    if use_cache and abs(scale - 1.0) < 1e-6:
        cached = CACHE_DIR / f"{image_id}.png"
        if cached.exists():
            img = cv2.imread(str(cached), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
    path = image_path(image_id)
    if path is None:
        return np.full((cfg.line_height, cfg.min_line_width), 235, np.uint8)
    out = normalize_line(read_gray(path), cfg, scale)
    if use_cache and abs(scale - 1.0) < 1e-6:
        cv2.imwrite(str(CACHE_DIR / f"{image_id}.png"), out)
    return out


def to_pil(gray: np.ndarray):
    return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))

# %% [markdown]
# ### 6.1 Warm the cache
#
# Normalisation costs ~15 ms per image and is deterministic, so it is done once for
# all 5 472 crops. Training then reads small PNGs, which keeps the dataloader ahead
# of an 8B model even at batch 16.

# %%
from concurrent.futures import ThreadPoolExecutor

def warm_cache(ids: Sequence[str], workers: int = 12) -> None:
    todo = [i for i in ids if not (CACHE_DIR / f"{i}.png").exists()]
    if not todo:
        print(f"[cache] {CACHE_DIR.name}: complete")
        return
    print(f"[cache] normalising {len(todo)} images -> {CACHE_DIR}")
    t0 = time.time()
    with ThreadPoolExecutor(workers) as ex:
        list(progress(ex.map(lambda i: load_line(i), todo), total=len(todo),
                      desc="cache", unit="img"))
    print(f"[cache] done in {time.time()-t0:.0f}s")


stage("image normalisation cache")
warm_cache(list(train_df.ID) + list(test_df.ID))

_sizes = []
for i in train_df.ID.sample(400, random_state=0):
    im = load_line(i)
    _sizes.append((im.shape[1], im.shape[0], (im.shape[1] // CFG.patch_grid) * (im.shape[0] // CFG.patch_grid)))
_sizes = np.array(_sizes)
print(f"[cache] normalised width  median {np.median(_sizes[:,0]):.0f} p90 {np.percentile(_sizes[:,0],90):.0f}")
print(f"[cache] normalised height median {np.median(_sizes[:,1]):.0f}")
print(f"[cache] visual tokens     median {np.median(_sizes[:,2]):.0f} p90 {np.percentile(_sizes[:,2],90):.0f} max {_sizes[:,2].max()}")

fig, axes = plt.subplots(4, 1, figsize=(16, 6))
for ax_, (_, row) in zip(axes, train_df.sample(4, random_state=11).iterrows()):
    ax_.imshow(load_line(row.ID), cmap="gray")
    ax_.set_title(row.Target, fontsize=8, loc="left")
    ax_.axis("off")
plt.tight_layout()
plt.savefig(OUT / "eda_normalised.png", dpi=110)
plt.close(fig)
print(f"[cache] preview -> {OUT/'eda_normalised.png'}")

# %% [markdown]
# ### 6.2 Augmentation
#
# Handwriting augmentation has to change *how* a glyph is written without changing
# *which* glyph it is. The stack below is deliberately mild on geometry (±1.5°
# rotation, ±0.06 shear) and more aggressive on the degradation axes that actually
# vary across the archive: ink weight, paper contrast, blur, noise and stains.

# %%
def _rand_affine(img: np.ndarray, rng: np.random.Generator, cfg: Config) -> np.ndarray:
    h, w = img.shape[:2]
    angle = rng.uniform(-cfg.aug_rotate_deg, cfg.aug_rotate_deg)
    shear = rng.uniform(-cfg.aug_shear, cfg.aug_shear)
    sx = 1.0 + rng.uniform(-cfg.aug_scale_jitter, cfg.aug_scale_jitter)
    sy = 1.0 + rng.uniform(-cfg.aug_scale_jitter, cfg.aug_scale_jitter) * 0.5
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    M[0, 0] *= sx
    M[0, 1] += shear
    M[1, 1] *= sy
    M[0, 2] += rng.uniform(-2, 2)
    M[1, 2] += rng.uniform(-2, 2)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _elastic(img: np.ndarray, rng: np.random.Generator, alpha=1.6, sigma=0.12) -> np.ndarray:
    """Low-frequency wobble along the baseline - mimics natural hand tremor."""
    h, w = img.shape[:2]
    gh, gw = 3, max(4, w // max(h, 1) + 2)
    dy = rng.normal(0, alpha, (gh, gw)).astype(np.float32) * h * sigma * 0.25
    dx = rng.normal(0, alpha, (gh, gw)).astype(np.float32) * h * sigma * 0.15
    dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_CUBIC)
    dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_CUBIC)
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float32)
    f = f * rng.uniform(0.85, 1.15) + rng.uniform(-18, 18)          # brightness/contrast
    f = np.clip(f, 0, 255)
    gamma = rng.uniform(0.75, 1.35)
    f = 255.0 * np.power(f / 255.0, gamma)
    if rng.random() < 0.35:                                          # illumination ramp
        w = img.shape[1]
        ramp = np.linspace(rng.uniform(-15, 15), rng.uniform(-15, 15), w, dtype=np.float32)
        f = f + ramp[None, :]
    return np.clip(f, 0, 255).astype(np.uint8)


def _morph(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    k = np.ones((2, 2), np.uint8) if rng.random() < 0.7 else np.ones((3, 3), np.uint8)
    # ink is dark: erode() thickens strokes, dilate() thins them
    return cv2.erode(img, k) if rng.random() < 0.5 else cv2.dilate(img, k)


def _noise(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = img.astype(np.float32)
    if rng.random() < 0.6:
        f += rng.normal(0, rng.uniform(2, 9), img.shape).astype(np.float32)
    if rng.random() < 0.5:
        f = cv2.GaussianBlur(f, (0, 0), rng.uniform(0.4, 1.3))
    if rng.random() < 0.25:                                          # JPEG ringing
        q = int(rng.integers(35, 85))
        ok, enc = cv2.imencode(".jpg", np.clip(f, 0, 255).astype(np.uint8),
                               [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            f = cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    return np.clip(f, 0, 255).astype(np.uint8)


def _stains(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(int(rng.integers(1, 4))):
        r = int(rng.integers(max(3, h // 12), max(5, h // 3)))
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        layer = np.zeros_like(out, dtype=np.float32)
        cv2.circle(layer, (cx, cy), r, float(rng.uniform(18, 55)), -1)
        layer = cv2.GaussianBlur(layer, (0, 0), r * 0.6)
        out = np.clip(out.astype(np.float32) - layer, 0, 255).astype(np.uint8)
    return out


def augment_line(img: np.ndarray, rng: np.random.Generator, cfg: Config = None) -> np.ndarray:
    cfg = cfg or CFG
    if rng.random() > cfg.aug_prob:
        return img
    out = img
    out = _rand_affine(out, rng, cfg)
    if rng.random() < cfg.aug_elastic_prob:
        out = _elastic(out, rng)
    if rng.random() < cfg.aug_photometric_prob:
        out = _photometric(out, rng)
    if rng.random() < cfg.aug_morph_prob:
        out = _morph(out, rng)
    if rng.random() < cfg.aug_noise_prob:
        out = _noise(out, rng)
    if rng.random() < cfg.aug_cutout_prob:
        out = _stains(out, rng)
    return out


_rng = np.random.default_rng(0)
_demo_id = train_df.ID.iloc[5]
_base = load_line(_demo_id)
fig, axes = plt.subplots(6, 1, figsize=(16, 8))
axes[0].imshow(_base, cmap="gray"); axes[0].set_title("normalised (no augmentation)", fontsize=8, loc="left")
for ax_ in axes[1:]:
    ax_.imshow(augment_line(_base, _rng), cmap="gray")
for ax_ in axes:
    ax_.axis("off")
plt.tight_layout()
plt.savefig(OUT / "aug_preview.png", dpi=110)
plt.close(fig)
print(f"[aug] preview -> {OUT/'aug_preview.png'}")
