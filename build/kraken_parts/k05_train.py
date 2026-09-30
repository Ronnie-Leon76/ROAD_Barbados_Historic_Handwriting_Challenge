# %% [markdown]
# ## 5. Fine-tune PP-OCRv6
#
# ketos `path` format: `<id>.png` next to `<id>.gt.txt`. The whole crop is one
# bbox line — the starter kit's page segmenter is the wrong tool here.
#
# `--resize new` rebuilds the CTC head on the challenge alphabet. The pretrained
# 44-language softmax would spend most of its mass on scripts that never appear
# in Barbados secretary hand, and (ketos docs) "rapidly unlearn missing labels"
# if left in place. `--freeze-backbone` for the first ~2 epochs lets that new
# head settle before the visual tower moves. `--schedule cosine` matches the
# original PP-OCRv6 recipe.
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


def write_line_gt(ids: Sequence[str], texts: Sequence[str] | None,
                  dest: Path, mode: str = "identity") -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    texts = list(texts) if texts is not None else [None] * len(ids)
    paths = []
    for image_id, text in zip(ids, texts):
        png = dest / f"{image_id}.png"
        if not png.exists():
            img = load_line(image_id)
            if mode != "identity":
                img = photometric_variant(img, mode)
            cv2.imwrite(str(png), img)
        if text is not None:
            # ketos trains on NFD; we score NFC. Both are lossless for this charset.
            (dest / f"{image_id}.gt.txt").write_text(
                unicodedata.normalize("NFD", text), encoding="utf-8")
        paths.append(png)
    return paths


def write_manifest(paths: Sequence[Path], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(Path(p).resolve()) for p in paths) + "\n", encoding="utf-8")
    return path


def width_report(paths: Sequence[Path], height: int) -> dict:
    widths = []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
        widths.append(w * height / max(h, 1))
    widths = np.asarray(widths) if len(widths) else np.zeros(1)
    auto = int(math.ceil((widths.max() + 64) / 64.0) * 64)
    limit = CFG.max_line_width or auto
    return {"n": int(len(widths)), "max": int(widths.max()),
            "p99": float(np.percentile(widths, 99)), "max_width": int(limit),
            "over_limit": int((widths > limit).sum())}


def ketos_bin() -> str:
    found = shutil.which("ketos")
    if found:
        return found
    cand = Path(sys.executable).parent / ("ketos.exe" if os.name == "nt" else "ketos")
    if cand.exists():
        return str(cand)
    raise FileNotFoundError("ketos not on PATH - re-run the install cell")


def ketos_train_cmd(base: Path, train_m: Path, val_m: Path, out_dir: Path,
                    max_width: int) -> list:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = "bf16-mixed" if (torch.cuda.is_available()
                                 and torch.cuda.is_bf16_supported()) else "32-true"
    cmd = [ketos_bin(), "-v", "-d", device, "--precision", precision,
           "--workers", str(min(8, max(1, CFG.dataloader_workers))),
           "-s", str(CFG.seed),
           "train", "-f", "path", "-i", str(base), "--resize", CFG.resize,
           "-u", "NFD", "--normalize-whitespace",
           "-t", str(train_m), "-e", str(val_m), "-o", str(out_dir),
           "-B", str(CFG.batch_size), "-N", str(CFG.epochs),
           "-q", CFG.quit, "-F", "1",
           "--lrate", str(CFG.lr), "--warmup", str(CFG.warmup),
           "--schedule", CFG.schedule, "--optimizer", CFG.optimizer,
           "--max-width", str(max_width)]
    if CFG.freeze_backbone:
        cmd += ["--freeze-backbone", str(CFG.freeze_backbone)]
    if CFG.quit == "early" and CFG.min_epochs:
        cmd += ["--min-epochs", str(CFG.min_epochs)]
    cmd.append("--augment" if CFG.augment else "--no-augment")
    return cmd


FLAG_WITH_VALUE = {"--min-epochs", "--schedule", "--optimizer", "--freeze-backbone"}
FLAG_BARE = {"--normalize-whitespace"}


def slim_ketos(cmd: list) -> list:
    """Drop optional flags that some kraken 7.x builds do not recognise."""
    out, skip = [], False
    for tok in cmd:
        if skip:
            skip = False
            continue
        if tok in FLAG_BARE | FLAG_WITH_VALUE:
            skip = tok in FLAG_WITH_VALUE
            continue
        out.append(tok)
    return out


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


def run_logged(cmd: list, log_path: Path, echo_every: float = 30.0) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[ketos] $", " ".join(map(str, cmd))[:500])
    last = 0.0
    with open(log_path, "a", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace")
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
all_paths = write_line_gt(train_df.ID.tolist(), train_df.Target.tolist(), LINES)
widths = width_report(all_paths, CFG.line_height)
print(f"[data] {widths['n']} lines at h={CFG.line_height}: max {widths['max']}px "
      f"p99 {widths['p99']:.0f} -> --max-width {widths['max_width']}")
if widths["over_limit"]:
    print(f"!! {widths['over_limit']} lines exceed max-width and will be DROPPED")

BASE = download_base(CFG.model_id, CFG.model_file)
SMALL_BASE = download_base(CFG.small_model_id, CFG.small_model_file) if CFG.also_train_small else None

# %%
REPORT: dict = {}
DECODES: list[dict] = []


def train_one(tag: str, base: Path, tr_df: pd.DataFrame, va_df: pd.DataFrame) -> Path:
    out_dir = MODELS / tag
    weights = best_weights(out_dir)
    if weights is not None:
        print(f"[train] {tag}: reusing {weights.name}")
        return weights
    t0 = time.time()
    tr_m = write_manifest([LINES / f"{i}.png" for i in tr_df.ID],
                          KRAKEN_ROOT / "manifests" / f"{tag}_train.txt")
    va_m = write_manifest([LINES / f"{i}.png" for i in va_df.ID],
                          KRAKEN_ROOT / "manifests" / f"{tag}_val.txt")
    out_dir.mkdir(parents=True, exist_ok=True)
    (LOGS / f"train_{tag}.log").unlink(missing_ok=True)
    cmd = ketos_train_cmd(base, tr_m, va_m, out_dir, widths["max_width"])
    rc = run_logged(cmd, LOGS / f"train_{tag}.log")
    if rc != 0:
        # kraken minor releases drop/rename optional flags; retry a slimmer command.
        slim = slim_ketos(cmd)
        print(f"[train] {tag}: retry without optional flags")
        rc = run_logged(slim, LOGS / f"train_{tag}.log")
    weights = best_weights(out_dir)
    log = (LOGS / f"train_{tag}.log").read_text(encoding="utf-8", errors="replace")
    dropped = log.count("Failed. Replacing")
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
