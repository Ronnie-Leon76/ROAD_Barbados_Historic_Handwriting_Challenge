# %% [markdown]
# ## 4. Image pipeline
#
# Same cleaning as the VLM notebook (flatten, deskew, horizontal ink-crop, CLAHE)
# so a later ensemble is voting on reading, not on preprocessing. Height is 128 px
# — the native PP-OCRv6 medium input. Width is capped with slack for ketos
# padding and augmentation; ketos *replaces* anything over `--max-width` in the
# DataLoader, which tanks GPU utilisation.

# %%
import cv2
from PIL import Image

cv2.setNumThreads(0)
CACHE_DIR = OUT / "cache" / f"h{CFG.line_height}_w{CFG.max_line_width}"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def read_gray(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
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
    if b - a < gray.shape[0]:
        return gray
    return gray[:, a:b]


def target_size(w: int, h: int, cfg: Config, scale: float = 1.0) -> tuple[int, int]:
    s = min((cfg.line_height * scale) / max(h, 1), cfg.max_upscale)
    tw, th = w * s, h * s
    cap = cfg.max_line_width or 10 ** 6
    if tw > cap:
        k = cap / tw
        tw, th = tw * k, th * k
    th = max(int(round(th)), cfg.min_line_height)
    tw = max(int(round(tw)), cfg.min_line_width)
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
    interp = cv2.INTER_AREA if tw < w else cv2.INTER_CUBIC
    return cv2.resize(gray, (tw, th), interpolation=interp)


def photometric_variant(img: np.ndarray, mode: str) -> np.ndarray:
    if mode == "identity":
        return img
    if mode == "dark":
        f = np.clip(255.0 * np.power(img / 255.0, 1.35), 0, 255).astype(np.uint8)
        return cv2.erode(f, np.ones((2, 2), np.uint8))
    if mode == "light":
        f = np.clip(255.0 * np.power(img / 255.0, 0.78), 0, 255).astype(np.uint8)
        return cv2.dilate(f, np.ones((2, 2), np.uint8))
    if mode == "sharp":
        blur = cv2.GaussianBlur(img, (0, 0), 1.2)
        return np.clip(cv2.addWeighted(img, 1.6, blur, -0.6, 0), 0, 255).astype(np.uint8)
    raise ValueError(mode)


def load_line(image_id: str, cfg: Config = None, scale: float = 1.0,
              use_cache: bool = True) -> np.ndarray:
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


from concurrent.futures import ThreadPoolExecutor

stage("image cache")
todo = [i for i in list(train_df.ID) + list(test_df.ID)
        if not (CACHE_DIR / f"{i}.png").exists()]
if todo:
    with ThreadPoolExecutor(12) as ex:
        list(progress(ex.map(lambda i: load_line(i), todo), total=len(todo),
                      desc="cache", unit="img"))
print(f"[cache] {CACHE_DIR} ready")
