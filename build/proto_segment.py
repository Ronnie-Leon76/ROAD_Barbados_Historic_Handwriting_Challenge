"""Prototype: main-line band detection + word segmentation on the real crops."""
import sys
import cv2
import numpy as np
import pandas as pd


def ink_map(gray: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(gray, (0, 0), 1.0)
    bg = cv2.medianBlur(g, 31)
    norm = cv2.divide(g, np.maximum(bg, 1), scale=255)
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    thr = cv2.adaptiveThreshold(norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 41, 12)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return thr


def main_band(binary: np.ndarray, min_frac=0.35):
    h, w = binary.shape
    prof = binary.sum(axis=1).astype(np.float32) / 255.0
    k = max(3, int(round(h * 0.06)) | 1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, k), 0).ravel()
    if prof.max() <= 0:
        return 0, h
    centre = h // 2
    weights = np.exp(-0.5 * ((np.arange(h) - centre) / (0.28 * h)) ** 2)
    seed = int(np.argmax(prof * weights))
    thr = prof.max() * min_frac
    top = seed
    while top > 0 and prof[top - 1] > thr:
        top -= 1
    bot = seed
    while bot < h - 1 and prof[bot + 1] > thr:
        bot += 1
    return top, bot + 1


def column_runs(binary, band, rel_thr=0.04, smooth_frac=0.05):
    top, bot = band
    band_h = max(bot - top, 1)
    strip = binary[top:bot]
    col = strip.sum(axis=0).astype(np.float32) / 255.0
    k = max(3, int(round(band_h * smooth_frac)) | 1)
    col = cv2.GaussianBlur(col.reshape(1, -1), (k, 1), 0).ravel()
    on = col > max(rel_thr * band_h, 0.6)
    runs = []
    i, w = 0, len(on)
    while i < w:
        if on[i]:
            j = i
            while j < w and on[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    return runs, band_h


def split_to_n(runs, n_words):
    """Merge across the smallest gaps until exactly n_words segments remain."""
    if len(runs) < n_words:
        return None, 0.0
    segs = [list(r) for r in runs]
    merged_max = 0.0
    while len(segs) > n_words:
        gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
        idx = int(np.argmin(gaps))
        merged_max = max(merged_max, gaps[idx])
        segs[idx][1] = segs[idx + 1][1]
        del segs[idx + 1]
    if len(segs) > 1:
        sep_min = min(segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1))
    else:
        sep_min = 1e9
    margin = (sep_min + 1.0) / (merged_max + 1.0)
    return [tuple(s) for s in segs], margin


def main(n=400):
    df = pd.read_csv("Train.csv").sample(n, random_state=0)
    ok = 0
    margins = []
    fail_runs = 0
    for _, r in df.iterrows():
        img = cv2.imread(f"data/images/{r.ID}.jpg", cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        b = ink_map(img)
        band = main_band(b)
        words = str(r.Target).split()
        runs, band_h = column_runs(b, band)
        segs, margin = split_to_n(runs, len(words))
        if segs is None:
            fail_runs += 1
            continue
        margins.append(margin)
        if margin >= 1.6:
            ok += 1
    m = np.array(margins) if margins else np.array([0.0])
    print(f"n={len(df)}  too-few-runs={fail_runs}  usable(margin>=1.6)={ok} ({ok/len(df):.1%})")
    print("margin percentiles:", np.round(np.percentile(m, [10, 25, 50, 75, 90]), 2))
    for t in (1.3, 1.6, 2.0, 2.5, 3.0):
        print(f"  margin>={t}: {(m >= t).mean():.1%}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
