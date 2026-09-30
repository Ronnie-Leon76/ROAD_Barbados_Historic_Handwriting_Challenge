# %% [markdown]
# ## 5. Fine-tune PP-OCRv6
#
# ketos `path` format: `<id>.png` next to `<id>.gt.txt`. The whole crop is one
# bbox line — the starter kit's page segmenter is the wrong tool here.
#
# `--resize new` rebuilds the CTC head on the challenge alphabet. The pretrained
# 44-language softmax would spend most of its mass on scripts that never appear
# in Barbados secretary hand. ketos is invoked as `-q fixed --resize new` with
# no `--freeze-backbone`. Line PNGs are rewritten to fit padding + CTC stride
# so the DataLoader does not spend the epoch on `Failed. Replacing with sample`.
#
# Zero-shot CSVs and `best_*.safetensors` already on disk are reused. Wipe a
# fold directory only if that fold crashed.
#
# Every fold is trained. A 16 M model on 3 300 lines is minutes, not hours, and
# five independent hypotheses are the single biggest lever the VLM run left on
# the table (it trained fold 0 only).

# %%
from huggingface_hub import hf_hub_download

KRAKEN_ROOT = OUT / "kraken"
LINES = KRAKEN_ROOT / "lines"
MODELS = KRAKEN_ROOT / "models"
LOGS = KRAKEN_ROOT / "logs"
PREDS = KRAKEN_ROOT / "preds"
for d in (KRAKEN_ROOT, LINES, MODELS, LOGS, PREDS, KRAKEN_ROOT / "jobs"):
    d.mkdir(parents=True, exist_ok=True)

# NRTR pe is 2048 tokens, PP-OCRv6 width stride is 8. ketos pads 16 px/side.
PPOCRV6_NRTR_MAX_WIDTH = (2048 - 2) * 8  # 16368
PPOCRV6_DEFAULT_MAX_WIDTH = 2560
WIDTH_SUBSAMPLING = 8
KETOS_PAD = 16
UNSUPPORTED_FLAG_RE = re.compile(r"Option (--[\w-]+) is not supported")


def nfd_ws(text: str) -> str:
    t = unicodedata.normalize("NFD", str(text or ""))
    return re.sub(r"\s+", " ", t).strip()


def ctc_span(text: str) -> int:
    t = nfd_ws(text)
    if not t:
        return 1
    repeats = sum(a == b for a, b in zip(t, t[1:]))
    return len(t) + repeats


def ketos_usable_width(max_width: int | None = None) -> int:
    """PNG width that still fits after ketos pad (+ augment scale)."""
    cap = int(max_width or CFG.max_line_width or PPOCRV6_DEFAULT_MAX_WIDTH)
    inner = cap - 2 * KETOS_PAD
    if CFG.augment:
        inner = int(inner / 1.25)
    return max(inner, CFG.min_line_width)


def fit_for_ketos(img: np.ndarray, text: str | None, usable: int) -> np.ndarray:
    h, w = img.shape[:2]
    th = int(CFG.line_height)
    tw = max(int(round(w * th / max(h, 1))), int(CFG.min_line_width))
    if text:
        tw = max(tw, WIDTH_SUBSAMPLING * (ctc_span(text) + 1))
    tw = min(max(tw, 8), int(usable))
    if h == th and w == tw:
        return img
    interp = cv2.INTER_AREA if (tw < w or th < h) else cv2.INTER_CUBIC
    return cv2.resize(img, (tw, th), interpolation=interp)


def png_needs_refit(png: Path, usable: int) -> bool:
    if not png.exists() or png.stat().st_size < 32:
        return True
    with Image.open(png) as im:
        w, h = im.size
    return h != CFG.line_height or w > usable or w < 8


def write_line_gt(ids: Sequence[str], texts: Sequence[str] | None,
                  dest: Path, mode: str = "identity",
                  force: bool = False) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    texts = list(texts) if texts is not None else [None] * len(ids)
    usable = ketos_usable_width()
    paths = []
    for image_id, text in zip(ids, texts):
        png = dest / f"{image_id}.png"
        if force or png_needs_refit(png, usable):
            img = load_line(image_id)
            if mode != "identity":
                img = photometric_variant(img, mode)
            img = fit_for_ketos(img, text, usable)
            cv2.imwrite(str(png), img)
        if text is not None:
            (dest / f"{image_id}.gt.txt").write_text(nfd_ws(text), encoding="utf-8")
        paths.append(png)
    return paths


def sample_fits(png: Path, text: str, max_width: int) -> bool:
    t = nfd_ws(text)
    if not t or not png.exists():
        return False
    with Image.open(png) as im:
        w, h = im.size
    if w < 8 or h < 8:
        return False
    scaled_w = w * CFG.line_height / max(h, 1)
    if scaled_w + 2 * KETOS_PAD > max_width:
        return False
    if ctc_span(t) + 1 > scaled_w // WIDTH_SUBSAMPLING:
        return False
    return True


def usable_paths(df: pd.DataFrame, max_width: int) -> list[Path]:
    kept = []
    for image_id, text in zip(df.ID, df.Target):
        png = LINES / f"{image_id}.png"
        if sample_fits(png, text, max_width):
            kept.append(png)
    return kept


def write_manifest(paths: Sequence[Path], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(Path(p).resolve()) for p in paths) + "\n", encoding="utf-8")
    return path


def width_report(paths: Sequence[Path], height: int) -> dict:
    widths = []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
        if w < 1 or h < 1:
            continue
        widths.append(w * height / h)
    widths = np.asarray(widths, dtype=float) if len(widths) else np.array([PPOCRV6_DEFAULT_MAX_WIDTH], dtype=float)
    finite = widths[np.isfinite(widths)]
    if finite.size == 0:
        finite = np.array([PPOCRV6_DEFAULT_MAX_WIDTH], dtype=float)
    auto = int(math.ceil((float(finite.max()) + 64) / 64.0) * 64)
    limit = CFG.max_line_width or auto
    if not math.isfinite(limit) or limit < 1:
        limit = PPOCRV6_DEFAULT_MAX_WIDTH
    limit = int(min(limit, PPOCRV6_NRTR_MAX_WIDTH))
    return {"n": int(len(widths)), "max": int(finite.max()),
            "p99": float(np.percentile(finite, 99)), "max_width": limit,
            "over_limit": int((finite > limit).sum())}


def ketos_bin() -> str:
    found = shutil.which("ketos")
    if found:
        return found
    cand = Path(sys.executable).parent / ("ketos.exe" if os.name == "nt" else "ketos")
    if cand.exists():
        return str(cand)
    raise FileNotFoundError("ketos not on PATH - re-run the install cell")


TRAIN_ENV = {
    "TORCH_COMPILE_DISABLE": "1",
    "TORCHDYNAMO_DISABLE": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}


def ketos_train_cmd(base: Path, train_m: Path, val_m: Path, out_dir: Path,
                    max_width: int, batch_size: int | None = None) -> list:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = "bf16-mixed" if (torch.cuda.is_available()
                                 and torch.cuda.is_bf16_supported()) else "32-true"
    quit = "fixed" if CFG.quit == "early" else (CFG.quit or "fixed")
    if quit != CFG.quit:
        print(f"[train] ppocrv6 cannot use -q {CFG.quit} (kraken#792); "
              f"using -q {quit} -N {CFG.epochs}")
    try:
        max_w = int(max_width)
    except (TypeError, OverflowError, ValueError):
        max_w = PPOCRV6_DEFAULT_MAX_WIDTH
    if not math.isfinite(max_w) or max_w < 1:
        max_w = PPOCRV6_DEFAULT_MAX_WIDTH
    hard = int(CFG.max_line_width or PPOCRV6_DEFAULT_MAX_WIDTH)
    max_w = min(max_w, hard, PPOCRV6_NRTR_MAX_WIDTH)
    batch = max(1, int(batch_size or CFG.batch_size))
    cmd = [ketos_bin(), "-v", "-d", device, "--precision", precision,
           "--workers", str(max(0, min(8, CFG.dataloader_workers))),
           "-s", str(CFG.seed),
           "train", "-f", "path", "-i", str(base), "--resize", CFG.resize,
           "-u", "NFD", "--normalize-whitespace",
           "-t", str(train_m), "-e", str(val_m), "-o", str(out_dir),
           "-B", str(batch), "-N", str(CFG.epochs),
           "-q", quit, "-F", "1",
           "--lrate", str(CFG.lr), "--warmup", str(CFG.warmup),
           "--max-width", str(max_w)]
    if CFG.schedule:
        cmd += ["--schedule", CFG.schedule]
    if CFG.optimizer:
        cmd += ["--optimizer", CFG.optimizer]
    cmd.append("--augment" if CFG.augment else "--no-augment")
    return cmd


FLAG_WITH_VALUE = {"--min-epochs", "--schedule", "--optimizer", "--freeze-backbone"}
FLAG_BARE = {"--normalize-whitespace"}


def drop_ketos_flag(cmd: list, flag: str) -> list:
    """Remove one CLI flag (and its value, if it takes one)."""
    out, skip = [], False
    for tok in cmd:
        if skip:
            skip = False
            continue
        if tok == flag:
            skip = flag in FLAG_WITH_VALUE and flag not in FLAG_BARE
            continue
        out.append(tok)
    return out


def force_quit_fixed(cmd: list) -> list:
    out = list(cmd)
    if "-q" in out:
        out[out.index("-q") + 1] = "fixed"
    return drop_ketos_flag(out, "--min-epochs")


def best_weights(out_dir: Path) -> Path | None:
    found = []
    for p in Path(out_dir).glob("best_*.safetensors"):
        try:
            found.append((float(p.stem.split("_", 1)[1]), p))
        except ValueError:
            continue
    if found:
        return max(found)[1]
    safes = sorted(Path(out_dir).glob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    return safes[-1] if safes else None


def run_logged(cmd: list, log_path: Path, echo_every: float = 30.0,
               extra_env: dict | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[ketos] $", " ".join(map(str, cmd))[:500])
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    last = 0.0
    with open(log_path, "a", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", env=env)
        for line in proc.stdout:
            fh.write(line)
            now = time.time()
            if INTERACTIVE or now - last > echo_every or "error" in line.lower():
                print(f"  {line.rstrip()[:220]}")
                last = now
        return proc.wait()


WORKER = r'''
import json, math, sys
from pathlib import Path
from PIL import Image
from kraken.containers import BBoxLine, Segmentation
from kraken.configs import RecognitionInferenceConfig
from kraken.tasks import RecognitionTaskModel

job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model = RecognitionTaskModel.load_model(job["model"])
dev = job.get("device", "cpu")
if str(dev).startswith("cuda"):
    accelerator, devices = "gpu", [int(dev.split(":")[1]) if ":" in str(dev) else 0]
else:
    accelerator, devices = "cpu", 1
config = RecognitionInferenceConfig(batch_size=job["batch"], num_line_workers=0,
                                    accelerator=accelerator, device=devices)
items, gap, out = job["items"], 16, []
for start in range(0, len(items), job["batch"]):
    chunk = items[start:start + job["batch"]]
    ims = [Image.open(it["path"]).convert("L") for it in chunk]
    width = max(im.width for im in ims)
    height = sum(im.height for im in ims) + gap * (len(ims) + 1)
    canvas = Image.new("L", (width, height), 255)
    lines, y = [], gap
    for k, im in enumerate(ims):
        canvas.paste(im, (0, y))
        lines.append(BBoxLine(id=f"l{k}", bbox=(0, y, im.width, y + im.height)))
        y += im.height + gap
    seg = Segmentation(type="bbox", imagename="canvas", text_direction="horizontal-lr",
                       script_detection=False, lines=lines)
    recs = list(model.predict(im=canvas, segmentation=seg, config=config))
    for it, rec in zip(chunk, recs):
        conf = [max(float(c), 1e-6) for c in (getattr(rec, "confidences", None) or [])]
        score = sum(math.log(c) for c in conf) / len(conf) if conf else -5.0
        pred = rec.prediction if rec.prediction is not None else ""
        out.append({"ID": it["id"], "text": pred, "score": score})
    print(f"decode {min(start + len(chunk), len(items))}/{len(items)}", flush=True)
Path(job["out"]).write_text(json.dumps(out), encoding="utf-8")
'''


def decode_model(model: Path, ids: Sequence[str], tag: str) -> pd.DataFrame:
    """Identity + photometric TTA. Each view is one hypothesis with a CTC confidence."""
    frames = []
    worker = KRAKEN_ROOT / "worker.py"
    worker.write_text(WORKER, encoding="utf-8")
    for mode in CFG.tta_modes:
        dest = LINES if mode == "identity" else KRAKEN_ROOT / f"lines_{mode}"
        paths = write_line_gt(ids, None, dest, mode=mode)
        job = KRAKEN_ROOT / "jobs" / f"{tag}_{mode}.json"
        res = KRAKEN_ROOT / "jobs" / f"{tag}_{mode}.out.json"
        job.write_text(json.dumps({
            "model": str(model), "batch": int(CFG.infer_batch), "out": str(res),
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "items": [{"id": i, "path": str(p.resolve())} for i, p in zip(ids, paths)],
        }), encoding="utf-8")
        rc = run_logged([sys.executable, str(worker), str(job)], LOGS / f"decode_{tag}.log")
        if rc != 0 or not res.exists():
            raise RuntimeError(f"decode {tag}/{mode} failed rc={rc}")
        df = pd.DataFrame(json.loads(res.read_text(encoding="utf-8")))
        df["view"] = f"{tag}:{mode}"
        df["text"] = df["text"].fillna("").map(
            lambda s: normalize_text(unicodedata.normalize("NFC", s)))
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def top1_score(df: pd.DataFrame, ids: Sequence[str]) -> dict:
    ident = (df[df.view.str.endswith(":identity")]
             if "view" in df.columns else df).drop_duplicates("ID").set_index("ID")["text"]
    refs = train_df.set_index("ID").loc[list(ids), "Target"].tolist()
    return competition_score(refs, [ident.get(i, "") for i in ids])


def download_base(model_id: str, filename: str) -> Path:
    path = Path(hf_hub_download(model_id, filename, local_dir=str(KRAKEN_ROOT / "base")))
    print(f"[model] {model_id} -> {path}")
    return path


stage("prepare line ground truth")
all_paths = write_line_gt(train_df.ID.tolist(), train_df.Target.tolist(), LINES, force=True)
widths = width_report(all_paths, CFG.line_height)
n_skip = sum(1 for i, t in zip(train_df.ID, train_df.Target)
             if not sample_fits(LINES / f"{i}.png", t, CFG.max_line_width))
print(f"[data] {widths['n']} lines at h={CFG.line_height}: max {widths['max']}px "
      f"p99 {widths['p99']:.0f} -> --max-width {widths['max_width']} "
      f"(png cap {ketos_usable_width()}px, omit {n_skip})")

BASE = download_base(CFG.model_id, CFG.model_file)
SMALL_BASE = download_base(CFG.small_model_id, CFG.small_model_file) if CFG.also_train_small else None

# %%
REPORT: dict = {}
DECODES: list[dict] = []


def _free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _is_oom(log: str) -> bool:
    low = log.lower()
    return "out of memory" in low or "outofmemoryerror" in low


def train_one(tag: str, base: Path, tr_df: pd.DataFrame, va_df: pd.DataFrame) -> Path:
    out_dir = MODELS / tag
    weights = best_weights(out_dir)
    if weights is not None:
        print(f"[train] {tag}: reusing {weights.name}")
        return weights
    t0 = time.time()
    write_line_gt(list(tr_df.ID) + list(va_df.ID),
                  list(tr_df.Target) + list(va_df.Target), LINES)
    tr_paths = usable_paths(tr_df, CFG.max_line_width)
    va_paths = usable_paths(va_df, CFG.max_line_width)
    skipped_tr = len(tr_df) - len(tr_paths)
    skipped_va = len(va_df) - len(va_paths)
    if skipped_tr or skipped_va:
        print(f"[train] {tag}: omitting {skipped_tr} train / {skipped_va} val lines that cannot fit CTC")
    if len(tr_paths) < 16 or len(va_paths) < 4:
        raise RuntimeError(f"{tag}: too few usable lines train={len(tr_paths)} val={len(va_paths)}")
    tr_m = write_manifest(tr_paths, KRAKEN_ROOT / "manifests" / f"{tag}_train.txt")
    va_m = write_manifest(va_paths, KRAKEN_ROOT / "manifests" / f"{tag}_val.txt")
    out_dir.mkdir(parents=True, exist_ok=True)
    (LOGS / f"train_{tag}.log").unlink(missing_ok=True)
    max_w = int(CFG.max_line_width or PPOCRV6_DEFAULT_MAX_WIDTH)
    batches, b = [], max(1, int(CFG.batch_size))
    while b not in batches:
        batches.append(b)
        b = max(1, b // 2)
    rc, log_txt = 1, ""
    for batch in batches:
        _free_gpu()
        cmd = ketos_train_cmd(base, tr_m, va_m, out_dir, max_w, batch_size=batch)
        print(f"[train] {tag}: -B {batch} --max-width {max_w}")
        rc = run_logged(cmd, LOGS / f"train_{tag}.log", extra_env=TRAIN_ENV)
        log_txt = (LOGS / f"train_{tag}.log").read_text(encoding="utf-8", errors="replace")
        if rc == 0:
            break
        hit = UNSUPPORTED_FLAG_RE.search(log_txt)
        if hit:
            cmd = drop_ketos_flag(cmd, hit.group(1))
            print(f"[train] {tag}: retry without {hit.group(1)}")
            rc = run_logged(cmd, LOGS / f"train_{tag}.log", extra_env=TRAIN_ENV)
            log_txt = (LOGS / f"train_{tag}.log").read_text(encoding="utf-8", errors="replace")
            if rc == 0:
                break
        if "cannot convert float infinity to integer" in log_txt:
            cmd = force_quit_fixed(cmd)
            print(f"[train] {tag}: retry with -q fixed (kraken#792)")
            rc = run_logged(cmd, LOGS / f"train_{tag}.log", extra_env=TRAIN_ENV)
            log_txt = (LOGS / f"train_{tag}.log").read_text(encoding="utf-8", errors="replace")
            if rc == 0:
                break
        if _is_oom(log_txt):
            print(f"[train] {tag}: CUDA OOM at batch {batch}; trying smaller")
            continue
        break
    weights = best_weights(out_dir)
    dropped = log_txt.count("Failed. Replacing")
    if dropped:
        print(f"!! {tag}: ketos replaced {dropped} unusable samples")
    if rc != 0 or weights is None:
        raise RuntimeError(f"ketos {tag} failed rc={rc}; see {LOGS / f'train_{tag}.log'}")
    print(f"[train] {tag}: {weights.name} in {fmt_hms(time.time() - t0)}")
    return weights


def run_family(name: str, base: Path, family: str) -> list[dict]:
    out = []
    stage(f"zero-shot {name}")
    zs_ids = train_df.ID.tolist()[: min(800, len(train_df))]
    zs_path = PREDS / f"zeroshot_{name}.csv"
    if zs_path.exists():
        zs = pd.read_csv(zs_path, keep_default_na=False)
    else:
        zs = decode_model(base, zs_ids, f"zs_{name}")
        zs.to_csv(zs_path, index=False)
    REPORT[f"zeroshot_{name}"] = top1_score(zs, zs_ids)
    print(f"[zero-shot] {name}: score {REPORT[f'zeroshot_{name}']['score']:.5f} "
          f"CER {REPORT[f'zeroshot_{name}']['cer']:.4f} WER {REPORT[f'zeroshot_{name}']['wer']:.4f}")
    zs = REPORT[f"zeroshot_{name}"]
    NOTIFY.send(f"zero-shot {name} {zs['score']:.5f}",
                f"{name} unscored-on-Barbados (zero-shot) "
                f"score {zs['score']:.5f} CER {zs['cer']:.4f} WER {zs['wer']:.4f}")

    stage(f"fine-tune {name} on {len(CFG.train_folds)} fold(s)")
    for fold in progress(CFG.train_folds, desc=name, unit="fold"):
        tag = f"{name}_fold{fold}"
        tr_df, va_df = fold_frames(fold)
        weights = train_one(tag, base, tr_df, va_df)
        oof_path, test_path = PREDS / f"oof_{tag}.csv", PREDS / f"test_{tag}.csv"
        if oof_path.exists() and test_path.exists():
            oof = pd.read_csv(oof_path, keep_default_na=False)
            test = pd.read_csv(test_path, keep_default_na=False)
        else:
            oof = decode_model(weights, va_df.ID.tolist(), f"{tag}_oof")
            test = decode_model(weights, test_df.ID.tolist(), f"{tag}_test")
            oof.to_csv(oof_path, index=False)
            test.to_csv(test_path, index=False)
        res = top1_score(oof, va_df.ID.tolist())
        REPORT[tag] = res
        print(f"[oof] {tag}: {res['score']:.5f}  CER {res['cer']:.4f}  WER {res['wer']:.4f}")
        NOTIFY.fold_done(tag, res["score"], res["cer"], res["wer"])
        out.append({"tag": tag, "family": family, "fold": fold,
                    "oof": oof, "test": test, "score": res["score"]})
    return out


DECODES = run_family("medium", BASE, "ctc_medium")
if SMALL_BASE is not None:
    DECODES += run_family("small", SMALL_BASE, "ctc_small")
