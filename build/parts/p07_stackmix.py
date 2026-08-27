# %% [markdown]
# ## 7. StackMix — word-crop recombination
#
# **Permitted by the organisers** (discussion #34514): segmenting crops out of the
# provided training images and programmatically recombining them into new labelled
# lines counts as training-data augmentation, as long as the process is fully
# automated and reproducible and no test images, external images or manual
# annotation are involved. Everything below reads only `Train.csv` images.
#
# ### Why the quality gates are strict
#
# The text is cursive and words are frequently ligatured, so a naive projection-profile
# split is wrong more often than it is right. Measured on 600 random training lines,
# an unfiltered splitter matches the true word count on ~27% of lines and
# under-segments badly on the rest. Two gates fix this:
#
# * **gap margin** — merge column runs until exactly `n_words` segments remain, then
#   require the smallest surviving separator to be ≥ `stackmix_margin`× the largest
#   gap that had to be merged. Ambiguous splits are rejected.
# * **width/length correlation** — the widths of the segments must correlate with the
#   character lengths of the words at ρ ≥ `stackmix_corr`. This catches the failure
#   mode where the band detector locks onto the wrong text line entirely.
#
# Together they accept ~7% of lines. That is a small bank, but it is a *clean* one,
# and a mislabelled synthetic line is far more damaging than a missing one.
#
# ### Closing the domain gap
#
# Real crops always carry slivers of the neighbouring lines and real parchment
# texture. A synthetic line pasted onto white paper would be trivially separable from
# the real distribution, so instead each synthetic line is composited into a **real
# background donor**: a training crop whose centre band has been inpainted away,
# keeping its paper, its stains, and its neighbouring-line fragments. Words are
# transferred by subtracting their ink darkness, which leaves no seams.

# %%
@dataclass
class WordCrop:
    img: np.ndarray      # grayscale patch covering the word plus vertical padding
    band_h: int          # height of the main-line band it came from
    core_h: float        # x-height of the source line (the real size reference)
    pad_top: int         # rows of padding above the band inside `img`
    pad_x: int           # columns of blank padding on each side of the ink
    stroke: float        # mean stroke width / x-height, a proxy for the hand


@dataclass
class Donor:
    bg: np.ndarray       # crop with the centre band inpainted away
    band: tuple          # (top, bottom) of the erased band
    core_h: float        # x-height of the line that was erased
    stroke: float        # stroke style of the scribe who wrote it
    gaps: np.ndarray     # inter-word gaps, as a fraction of the x-height


def line_style(binary: np.ndarray, band: tuple) -> tuple[float, float]:
    """(x-height, stroke width / x-height) of the main line.

    Band height is dominated by ascenders and descenders and is a poor size
    reference; the x-height - the rows carrying most of the ink - is what makes two
    hands look the same size. Stroke width comes from the mean length of horizontal
    ink runs, which separates a fine quill from a broad one.
    """
    top, bot = band
    strip = binary[top:bot] > 0
    if strip.size == 0 or not strip.any():
        return float(max(bot - top, 1)), 1.0
    rows = strip.sum(axis=1).astype(np.float32)
    core_h = float(max((rows > 0.5 * rows.max()).sum(), 1))
    ink = int(strip.sum())
    starts = int((strip[:, 1:] & ~strip[:, :-1]).sum() + strip[:, 0].sum())
    stroke = ink / max(starts, 1)
    return core_h, stroke / core_h


def main_line_band(binary: np.ndarray, min_frac: float = 0.35) -> tuple[int, int]:
    """Vertical extent of the centre line, ignoring neighbour slivers."""
    h = binary.shape[0]
    prof = binary.sum(axis=1).astype(np.float32) / 255.0
    k = max(3, int(round(h * 0.06)) | 1)
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (1, k), 0).ravel()
    if prof.max() <= 0:
        return 0, h
    weights = np.exp(-0.5 * ((np.arange(h) - h / 2) / (0.28 * h)) ** 2)
    seed = int(np.argmax(prof * weights))
    thr = prof.max() * min_frac
    top = seed
    while top > 0 and prof[top - 1] > thr:
        top -= 1
    bot = seed
    while bot < h - 1 and prof[bot + 1] > thr:
        bot += 1
    return top, bot + 1


def column_runs(binary: np.ndarray, band: tuple) -> list:
    top, bot = band
    band_h = max(bot - top, 1)
    col = binary[top:bot].sum(axis=0).astype(np.float32) / 255.0
    k = max(3, int(round(band_h * 0.05)) | 1)
    col = cv2.GaussianBlur(col.reshape(1, -1), (k, 1), 0).ravel()
    on = col > max(0.04 * band_h, 0.6)
    runs, i, w = [], 0, len(on)
    while i < w:
        if on[i]:
            j = i
            while j < w and on[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    return runs


def merge_to_n(runs: list, n: int) -> tuple[list | None, float, list]:
    """Merge across the smallest gaps until `n` segments remain."""
    if len(runs) < n or n < 1:
        return None, 0.0, []
    segs = [list(r) for r in runs]
    merged_max = 0.0
    while len(segs) > n:
        gaps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
        idx = int(np.argmin(gaps))
        merged_max = max(merged_max, gaps[idx])
        segs[idx][1] = segs[idx + 1][1]
        del segs[idx + 1]
    seps = [segs[i + 1][0] - segs[i][1] for i in range(len(segs) - 1)]
    margin = (min(seps) + 1.0) / (merged_max + 1.0) if seps else 1e9
    return [tuple(s) for s in segs], float(margin), seps


def segment_line(img: np.ndarray, words: Sequence[str], cfg: Config):
    """Return (segments, band, gaps) when the split passes both quality gates."""
    if len(words) < 3:
        return None
    binary = ink_mask(img)
    band = main_line_band(binary)
    if band[1] - band[0] < 8:
        return None
    segs, margin, seps = merge_to_n(column_runs(binary, band), len(words))
    if segs is None or margin < cfg.stackmix_margin:
        return None
    widths = np.array([b - a for a, b in segs], dtype=np.float64)
    lengths = np.array([len(w) for w in words], dtype=np.float64)
    if widths.std() < 1e-6 or lengths.std() < 1e-6:
        return None
    corr = float(np.corrcoef(widths, lengths)[0, 1])
    if not np.isfinite(corr) or corr < cfg.stackmix_corr:
        return None
    return segs, band, np.array(seps, dtype=np.float32)


def build_word_bank(df: pd.DataFrame, cfg: Config, max_lines: int | None = None):
    """Segment training lines into a bank of labelled word crops + background donors."""
    bank: dict[str, list[WordCrop]] = defaultdict(list)
    donors: list[Donor] = []
    rows = df if max_lines is None else df.head(max_lines)
    accepted = 0
    bar = progress(rows.itertuples(), total=len(rows), desc="stackmix scan", unit="line")
    for row in bar:
        path = image_path(row.ID)
        if path is None:
            continue
        # Segment at native resolution: the normalised copies are upscaled, which
        # blurs the inter-word gaps the splitter relies on.
        img = flatten_background(read_gray(path))
        if img.shape[0] < 36:
            continue
        words = row.Target.split()
        seg = segment_line(img, words, cfg)
        if seg is None:
            continue
        segs, band, gaps = seg
        accepted += 1
        bar.set_postfix_str(f"accepted {accepted}", refresh=False)
        top, bot = band
        band_h = bot - top
        binary = ink_mask(img)
        core_h, stroke = line_style(binary, band)
        pad = int(band_h * 0.3)
        y0, y1 = max(top - pad, 0), min(bot + pad, img.shape[0])
        m = max(1, int(band_h * 0.03))
        for (x0, x1), word in zip(segs, words):
            xa, xb = max(x0 - m, 0), min(x1 + m, img.shape[1])
            bank[word].append(WordCrop(img[y0:y1, xa:xb].copy(), band_h, core_h,
                                       top - y0, m, stroke))

        # Background donor: erase the centre band's ink, keep everything else.
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[top:bot] = binary[top:bot]
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        bg = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
        donors.append(Donor(bg, (top, bot), core_h, stroke, gaps / max(core_h, 1)))

    total_crops = sum(len(v) for v in bank.values())
    print(f"[stackmix] accepted {accepted}/{len(rows)} lines ({accepted/max(len(rows),1):.1%})")
    print(f"[stackmix] bank: {total_crops} crops over {len(bank)} word types, {len(donors)} donors")
    return dict(bank), donors


class StackMixRenderer:
    """Composites new labelled lines out of banked word crops and real backgrounds."""

    def __init__(self, bank: dict, donors: list, corpus: Sequence[str], cfg: Config,
                 aspect_per_char: np.ndarray | None = None):
        self.bank = bank
        self.donors = donors
        self.cfg = cfg
        # w / (h * n_chars) over the real crops. Donor lines are a biased sample -
        # the splitter only accepts well-spaced hands - so their gaps alone would
        # render systematically airier lines than the test set contains.
        self.apc = (np.asarray(aspect_per_char, dtype=np.float64)
                    if aspect_per_char is not None and len(aspect_per_char) else None)
        if self.apc is not None:
            self.apc = self.apc[np.isfinite(self.apc) & (self.apc > 0)]
        self.words = sorted(bank)
        self.word_set = set(self.words)
        freq = np.array([max(WORD_FREQ.get(w, 1), 1) for w in self.words], dtype=np.float64)
        self.unigram_p = freq / freq.sum()
        self.bigram: dict[str, list[str]] = defaultdict(list)
        for line in corpus:
            toks = line.split()
            for a, b in zip(toks, toks[1:]):
                if a in self.word_set and b in self.word_set:
                    self.bigram[a].append(b)
        self.corpus_lines = [ln.split() for ln in corpus]
        gaps = np.concatenate([d.gaps for d in donors if d.gaps.size]) if donors else np.array([0.4])
        self.gap_lo, self.gap_hi = np.percentile(gaps, [25, 75])
        self.gap_lo = max(float(self.gap_lo), 0.1)
        self.gap_hi = max(float(self.gap_hi), self.gap_lo + 0.1)
        self.nw_values, self.nw_p = self._length_prior()

    def _length_prior(self) -> tuple[np.ndarray, np.ndarray]:
        """Word counts drawn from the real lines, not from a flat range.

        The metric weights long references more heavily, so the useful synthetic
        lines are the long ones; sampling the empirical head of the distribution
        above `stackmix_min_words` puts the augmentation where the score is.
        """
        lo, hi = self.cfg.stackmix_min_words, self.cfg.stackmix_max_words
        counts = np.array([len(t) for t in self.corpus_lines], dtype=np.int64)
        counts = counts[(counts >= lo) & (counts <= hi)]
        if counts.size < 20:
            values = np.arange(lo, hi + 1)
            return values, np.full(values.size, 1.0 / values.size)
        values, freq = np.unique(counts, return_counts=True)
        return values, freq / freq.sum()

    # ----------------------------------------------------------------- text --
    def sample_words(self, rng: np.random.Generator) -> list[str]:
        n = int(rng.choice(self.nw_values, p=self.nw_p))
        if rng.random() < self.cfg.stackmix_ngram_prob:
            span = self._corpus_span(rng, n)
            if span:
                return span
        return self._markov(rng, n)

    def _corpus_span(self, rng: np.random.Generator, n: int) -> list[str] | None:
        for _ in range(12):
            toks = self.corpus_lines[int(rng.integers(len(self.corpus_lines)))]
            if len(toks) < 3:
                continue
            k = min(n, len(toks))
            s = int(rng.integers(0, len(toks) - k + 1))
            span = toks[s:s + k]
            if all(w in self.word_set for w in span):
                if len(span) < n:  # splice a second span from another line
                    tail = self._corpus_span(rng, n - len(span))
                    if tail:
                        span = span + tail
                return span
        return None

    def _markov(self, rng: np.random.Generator, n: int) -> list[str]:
        out = [self.words[int(rng.choice(len(self.words), p=self.unigram_p))]]
        while len(out) < n:
            succ = self.bigram.get(out[-1])
            if succ and rng.random() < 0.75:
                out.append(succ[int(rng.integers(len(succ)))])
            else:
                out.append(self.words[int(rng.choice(len(self.words), p=self.unigram_p))])
        return out

    # ---------------------------------------------------------------- image --
    def _background(self, donor: Donor, width: int, rng: np.random.Generator) -> np.ndarray:
        """Mirror-tile one donor horizontally so its neighbour lines stay coherent."""
        bg = donor.bg
        if bg.shape[1] >= width:
            off = int(rng.integers(0, bg.shape[1] - width + 1)) if bg.shape[1] > width else 0
            return bg[:, off:off + width].copy()
        reps = int(np.ceil(width / bg.shape[1])) + 1
        tiles = [bg if k % 2 == 0 else bg[:, ::-1] for k in range(reps)]
        return np.concatenate(tiles, axis=1)[:, :width].copy()

    def _pick_crop(self, word: str, style: float, rng: np.random.Generator) -> WordCrop | None:
        """Prefer a crop written in a hand similar to the donor's."""
        pool = self.bank.get(word)
        if not pool:
            return None
        if len(pool) <= 2:
            return pool[int(rng.integers(len(pool)))]
        order = np.argsort([abs(c.stroke - style) for c in pool])
        k = max(1, min(3, len(pool)))
        return pool[int(order[int(rng.integers(k))])]

    def render(self, rng: np.random.Generator) -> tuple[np.ndarray, str] | None:
        words = self.sample_words(rng)
        donor = self.donors[int(rng.integers(len(self.donors)))]
        crops = []
        for w in words:
            c = self._pick_crop(w, donor.stroke, rng)
            if c is None:
                return None
            crops.append(c)

        height = donor.bg.shape[0]
        band_top, band_bot = donor.band
        target_core = max(donor.core_h, 4.0)
        # Let the word count set the width, as it does on a real line, and cap only
        # at the widest aspect the corpus actually contains. Capping tighter than
        # this silently truncates the line and shortens every synthetic label.
        max_width = int(height * 34.0)

        # Scale every crop to the donor's x-height, then plan the layout on paper
        # before touching a pixel, so the whole line can be resized in one pass.
        iso = [(target_core / max(c.core_h, 1.0)) * float(rng.uniform(0.96, 1.04)) for c in crops]
        widths = [max(int(round(c.img.shape[1] * s)), 4) for c, s in zip(crops, iso)]
        pads = [c.pad_x * s for c, s in zip(crops, iso)]
        # Gaps are measured ink-to-ink, but each crop already carries blank padding,
        # so the padding of both neighbours has to come out of the gap.
        raw_gaps = [max(1.0, rng.uniform(self.gap_lo, self.gap_hi) * target_core
                        - pads[k] - pads[k + 1]) for k in range(len(crops) - 1)]
        margin = max(2, int(target_core * rng.uniform(0.1, 0.8)))

        # Give the line the width a real line of this many characters would have.
        # The donor pool is biased towards broad, generously spaced hands - those
        # are the ones the splitter can segment - so without this the synthetic set
        # sits to the right of the real aspect distribution and teaches the model a
        # geometry the test set does not contain. The correction is split between a
        # mild horizontal squeeze of the glyphs and the inter-word spacing; both are
        # clipped to ranges the real data already spans.
        xs, gap_scale = 1.0, 1.0
        if self.apc is not None:
            n_chars = len(" ".join(words))
            need = height * float(self.apc[int(rng.integers(self.apc.size))]) * n_chars - 2 * margin
            have = sum(widths) + sum(raw_gaps)
            xs = float(np.clip(need / max(have, 1.0), 0.82, 1.14))
            if raw_gaps:
                gap_scale = float(np.clip((need - sum(widths) * xs) / sum(raw_gaps), 0.4, 1.6))

        scaled = []
        for c, s in zip(crops, iso):
            nh = max(int(round(c.img.shape[0] * s)), 4)
            nw = max(int(round(c.img.shape[1] * s * xs)), 4)
            scaled.append((cv2.resize(c.img, (nw, nh), interpolation=cv2.INTER_LINEAR),
                           int(round(c.pad_top * s))))
        gaps = [max(1, int(round(g * gap_scale))) for g in raw_gaps]
        width = min(margin * 2 + sum(s[0].shape[1] for s in scaled) + sum(gaps), max_width)
        canvas = self._background(donor, int(width), rng)

        # One ink level for the whole line: a line is written with one pen, and
        # per-word darkness jitter is the giveaway that a crop is synthetic.
        ink_gain = float(rng.uniform(0.82, 1.0))
        x = margin
        used = 0
        for k, (im, pad_top) in enumerate(scaled):
            if x + im.shape[1] > canvas.shape[1] - 2:
                break
            if im.shape[0] >= height:
                im = im[:max(height - 1, 1)]
            y = int(np.clip(band_top - pad_top, 0, height - im.shape[0] - 1))
            crop_paper = float(np.percentile(im, 85))
            ink = np.clip(crop_paper - im.astype(np.float32), 0, 255)
            peak = float(ink.max())
            if peak > 20:                      # equalise contrast across scribes
                ink *= min(1.6, 150.0 / peak)
            region = canvas[y:y + im.shape[0], x:x + im.shape[1]].astype(np.float32)
            blended = np.clip(region - ink * ink_gain, 0, 255)
            canvas[y:y + im.shape[0], x:x + im.shape[1]] = blended.astype(np.uint8)
            x += im.shape[1] + (gaps[k] if k < len(gaps) else 0)
            used = k + 1

        if used < max(3, self.cfg.stackmix_min_words // 2):
            return None
        canvas = canvas[:, :int(np.clip(x + margin, 8, canvas.shape[1]))]
        if self.cfg.clahe:
            canvas = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(canvas)
        tw, th = target_size(canvas.shape[1], canvas.shape[0], self.cfg)
        interp = cv2.INTER_AREA if tw < canvas.shape[1] else cv2.INTER_CUBIC
        canvas = cv2.resize(canvas, (tw, th), interpolation=interp)
        return canvas, " ".join(words[:used])


# %%
STACKMIX = None
if CFG.stackmix:
    stage("StackMix word bank")
    set_seed(CFG.stackmix_seed)
    _bank, _donors = build_word_bank(train_df, CFG)
    if sum(len(v) for v in _bank.values()) >= 300 and len(_donors) >= 20:
        _apc = (tr_stats["w"] / (tr_stats["h"] * tr_stats["n_chars"].clip(lower=1))).values
        STACKMIX = StackMixRenderer(_bank, _donors, train_df["Target"].tolist(), CFG, _apc)
        rng = np.random.default_rng(CFG.stackmix_seed)
        fig, axes = plt.subplots(6, 1, figsize=(16, 8))
        shown = 0
        for ax_ in axes:
            out = None
            for _ in range(20):
                out = STACKMIX.render(rng)
                if out is not None:
                    break
            if out is None:
                ax_.axis("off")
                continue
            im, txt = out
            ax_.imshow(im, cmap="gray")
            ax_.set_title(txt, fontsize=8, loc="left")
            ax_.axis("off")
            shown += 1
        plt.tight_layout()
        plt.savefig(OUT / "stackmix_preview.png", dpi=110)
        plt.close(fig)
        print(f"[stackmix] rendered {shown} previews -> {OUT/'stackmix_preview.png'}")

        # Distribution check: synthetic geometry and label length must sit inside
        # the real ranges, otherwise the model learns a shortcut instead of a script.
        stats = []
        for _ in range(150):
            out = STACKMIX.render(rng)
            if out is not None:
                im, txt = out
                stats.append((im.shape[1] / max(im.shape[0], 1), len(txt), len(txt.split())))
        if stats:
            syn = np.array(stats)
            real_aspect = (tr_stats.w / tr_stats.h).values
            print(f"[stackmix] aspect  synthetic med {np.median(syn[:,0]):.1f} "
                  f"[p10 {np.percentile(syn[:,0],10):.1f}, p90 {np.percentile(syn[:,0],90):.1f}]"
                  f"  | real med {np.median(real_aspect):.1f} "
                  f"[p10 {np.percentile(real_aspect,10):.1f}, p90 {np.percentile(real_aspect,90):.1f}]")
            print(f"[stackmix] chars   synthetic med {np.median(syn[:,1]):.0f} "
                  f"| real med {train_df.Target.str.len().median():.0f}")
            print(f"[stackmix] words   synthetic med {np.median(syn[:,2]):.0f} "
                  f"| real med {train_df.Target.str.split().apply(len).median():.0f}")
        print("[stackmix] INSPECT THE PREVIEW before a long run. If the synthetic lines do "
              "not look like the real ones, set CFG.stackmix = False.")
    else:
        print("[stackmix] bank too small after quality gating - disabling")
        CFG.stackmix = False
else:
    print("[stackmix] disabled")
