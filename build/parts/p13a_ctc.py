# %% [markdown]
# ### 14.1 Second recogniser: kraken PP-OCRv6 (CTC)
#
# **Why this exists.** The BHL OCR leaderboard (finebooks/bhl-ocr-leaderboard,
# updated 2026-09-03) found that kraken's 15.9M-parameter PP-OCRv6 port reads
# 18th–19th-century book pages about as well as the best VLMs hundreds of times its
# size: reading-lane CER .034 [.024–.044] against .051 [.029–.077] for Qwen3.5-9B,
# .041 vs .067 diplomatic CER, and **0 % looped pages** against 1.5 %.
#
# Three things temper that for this challenge, and they shape how it is used here:
#
# * The board is **zero-shot on printed books**; this task is **fine-tuned on
#   17th-century handwriting**. A VLM fine-tuned on 3 300 in-domain lines is a
#   different animal from one read zero-shot, so the ranking may not transfer.
# * The reading-CER confidence intervals overlap. The board's own guidance is that
#   overlapping intervals mean "within noise". The differences that are clearly
#   outside noise are the *failure modes*: CTC never loops and never invents text
#   on near-empty lines, because it emits one label per image column.
# * Different architecture means different mistakes, and that is exactly what a
#   ROVER vote converts into accuracy.
#
# So the CTC model is **not a replacement** for the VLM. It is fine-tuned per fold on
# the same lines, decoded with photometric TTA, and handed to Section 15 as
#
# 1. extra hypotheses in the confusion network, and
# 2. an **anchor feature**: how close each VLM hypothesis is to what the CTC model
#    read, which penalises runaway or hallucinated VLM output.
#
# Section 15 tunes the fusion twice — with and without the CTC member — and keeps
# the CTC member only if it wins out-of-fold, the same gate GRPO uses. It also
# reports the zero-shot and fine-tuned CTC scores on *our* data, which is the
# direct test of whether the leaderboard finding carries over.
#
# **Isolation.** kraken 7.1 pins `torch>=2.9`, `scipy~=1.15` and `scikit-learn~=1.7`,
# which would fight the VLM stack. It therefore lives in its own virtualenv
# (`CFG.ctc_venv`) and is driven through subprocesses; nothing here can break the
# main environment. Any failure in this section prints a warning and the pipeline
# continues VLM-only.
#
# **Cost.** A 16M-parameter model on ~3 300 lines: a few minutes per fold on the
# card that just trained an 8B VLM, which is why every fold is trained by default
# (`CFG.ctc_folds = ()`), even when the VLM trains only one.

# %%
import dataclasses
import subprocess

import cv2
from PIL import Image

CTC_ROOT = OUT / "ctc"
CTC_LINES = CTC_ROOT / "lines"          # <ID>.png (+ <ID>.gt.txt for train)
CTC_MODELS = CTC_ROOT / "models"
CTC_LOGS = CTC_ROOT / "logs"


def ctc_config(cfg: Config = None) -> Config:
    """Image settings for the CTC branch: same cleaning, kraken's line height."""
    cfg = cfg or CFG
    return dataclasses.replace(cfg, line_height=cfg.ctc_line_height,
                               max_line_width=cfg.ctc_max_width or 8192,
                               max_upscale=max(cfg.max_upscale, 3.0))


def ctc_folds(cfg: Config = None) -> tuple:
    cfg = cfg or CFG
    return tuple(cfg.ctc_folds) if cfg.ctc_folds else tuple(range(cfg.n_folds))


def ctc_line(image_id: str, cfg: Config = None, mode: str = "identity") -> np.ndarray:
    """Background-flattened, deskewed line at the CTC model's input height.

    kraken height-normalises again internally; doing the cleaning here keeps the
    two recognisers looking at the same pixels, so their disagreements are about
    reading rather than preprocessing.
    """
    ccfg = ctc_config(cfg)
    path = image_path(image_id)
    if path is None:
        return np.full((ccfg.line_height, ccfg.min_line_width), 235, np.uint8)
    img = normalize_line(read_gray(path), ccfg)
    if mode != "identity":
        img = photometric_variant(img, mode)
    return img


def write_ctc_lines(ids: Sequence[str], texts: Sequence[str] | None,
                    dest: Path, cfg: Config = None, mode: str = "identity") -> list:
    """Write kraken `path`-format ground truth: X.png next to X.gt.txt."""
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    texts = list(texts) if texts is not None else [None] * len(ids)
    for image_id, text in zip(ids, texts):
        png = dest / f"{image_id}.png"
        if not png.exists():
            cv2.imwrite(str(png), ctc_line(image_id, cfg, mode))
        if text is not None:
            gt = dest / f"{image_id}.gt.txt"
            if not gt.exists() or gt.read_text(encoding="utf-8") != text:
                gt.write_text(text, encoding="utf-8")
        paths.append(png)
    return paths


def write_manifest(paths: Sequence[Path], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(str(Path(p).resolve()) for p in paths) + "\n",
                    encoding="utf-8")
    return path


def ctc_width_report(paths: Sequence[Path], cfg: Config = None) -> dict:
    """Widths *after* kraken rescales each line to the model height.

    ketos silently drops training lines wider than --max-width, and the width that
    matters is the one after its own height normalisation, not the PNG's. A crop
    that `normalize_line` had to shrink below 128 px to fit its width cap is scaled
    back up by kraken - and would vanish from training without this check.
    """
    cfg = cfg or CFG
    widths = []
    for p in paths:
        with Image.open(p) as im:
            w, h = im.size
        widths.append(w * cfg.ctc_line_height / max(h, 1))
    widths = np.asarray(widths) if widths else np.zeros(1)
    # +64: kraken pads 16 px each side before checking, plus augmentation slack.
    auto = int(math.ceil((widths.max() + 64) / 64.0) * 64)
    limit = cfg.ctc_max_width or auto
    return {"n": int(len(widths)), "max": int(widths.max()),
            "p99": float(np.percentile(widths, 99)), "max_width": int(limit),
            "over_limit": int((widths > limit).sum())}


def venv_bin(venv: Path, name: str) -> Path:
    sub = "Scripts" if os.name == "nt" else "bin"
    exe = f"{name}.exe" if os.name == "nt" else name
    return Path(venv) / sub / exe


def ketos_train_cmd(ketos: Path, base_model: Path, train_manifest: Path,
                    val_manifest: Path, out_dir: Path, cfg: Config = None,
                    max_width: int | None = None) -> list:
    """Fine-tune the PP-OCRv6 checkpoint. Every flag below exists in kraken 7.1.1.

    * `--resize union` keeps the pretrained codec and adds the challenge-only
      symbols (`^`, `~`, ...) as new output classes.
    * `-u NFD` matches how the base model was trained; predictions are folded back
      to NFC by `normalize_text` before scoring.
    * `--deterministic` is kraken's deterministic mode, which is `warn`-only for Lightning — the
      CUDA CTC-loss backward has no deterministic kernel and would otherwise abort.
    """
    cfg = cfg or CFG
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    precision = "bf16-mixed" if (torch.cuda.is_available()
                                 and torch.cuda.is_bf16_supported()) else "32-true"
    cmd = [str(ketos), "-v", "-d", device, "--precision", precision,
           "--workers", str(min(8, max(1, cfg.dataloader_workers))),
           "-s", str(cfg.seed)]
    if cfg.deterministic:
        cmd.append("--deterministic")
    cmd += ["train", "-f", "path", "-i", str(base_model), "--resize", "union",
            "-u", "NFD", "--normalize-whitespace",
            "-t", str(train_manifest), "-e", str(val_manifest),
            "-o", str(out_dir), "-B", str(cfg.ctc_batch_size),
            "-N", str(cfg.ctc_epochs), "-q", "fixed", "-F", "1",
            "--lrate", str(cfg.ctc_lr), "--warmup", str(cfg.ctc_warmup),
            "--max-width", str(max_width or cfg.ctc_max_width or 4096)]
    cmd.append("--augment" if cfg.ctc_augment else "--no-augment")
    return cmd


def best_ctc_weights(out_dir: Path) -> Path | None:
    """ketos writes best_<val_metric>.safetensors next to its checkpoints."""
    found = []
    for p in Path(out_dir).glob("best_*.safetensors"):
        try:
            found.append((float(p.stem.split("_", 1)[1]), p))
        except ValueError:
            continue
    return max(found)[1] if found else None


# The recogniser runs inside the kraken venv. Lines are tiled onto one tall canvas
# per batch so kraken sees a multi-line bbox segmentation and batches them through
# the network together, instead of one call per 16M-parameter forward pass.
CTC_WORKER = r'''
import json, math, sys
from pathlib import Path
from PIL import Image
from kraken.containers import BBoxLine, Segmentation
from kraken.configs import RecognitionInferenceConfig
from kraken.tasks import RecognitionTaskModel

job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
model = RecognitionTaskModel.load_model(job["model"])
dev = job.get("device", "cpu")
if dev.startswith("cuda"):
    accelerator, devices = "gpu", [int(dev.split(":")[1]) if ":" in dev else 0]
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
    for it, rec in zip(chunk, model.predict(im=canvas, segmentation=seg, config=config)):
        conf = [max(float(c), 1e-6) for c in (rec.confidences or [])]
        score = sum(math.log(c) for c in conf) / len(conf) if conf else -5.0
        out.append({"ID": it["id"], "text": rec.prediction, "score": score})
    print(f"ctc worker {min(start + len(chunk), len(items))}/{len(items)}", flush=True)
Path(job["out"]).write_text(json.dumps(out), encoding="utf-8")
'''


def run_logged(cmd: list, log_path: Path, env: dict | None = None,
               echo_every: float = 30.0) -> int:
    """Run a subprocess, tee its output to a log, echo a line every `echo_every` s."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("[ctc] $", " ".join(map(str, cmd))[:400])
    last = 0.0
    with open(log_path, "a", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, errors="replace", env=env)
        for line in proc.stdout:
            fh.write(line)
            now = time.time()
            if INTERACTIVE or now - last > echo_every or "error" in line.lower():
                print(f"[ctc]   {line.rstrip()[:200]}")
                last = now
        return proc.wait()


def ctc_frame_to_nbest(df: pd.DataFrame, model_tag: str) -> dict:
    """Rows of (ID, text, score, view) -> the pooled N-best structure of Section 14."""
    out: dict[str, list] = defaultdict(list)
    for row in df.itertuples():
        out[row.ID].append({"text": normalize_text(row.text), "score": float(row.score),
                            "view": row.view, "rank": 0,
                            "model": model_tag, "family": "ctc"})
    return dict(out)


def merge_nbest(base: dict, extra: dict, only: set | None = None) -> dict:
    merged = {k: list(v) for k, v in base.items()}
    for image_id, cands in extra.items():
        if only is not None and image_id not in only:
            continue
        merged.setdefault(image_id, []).extend(cands)
    return merged

# %%
CTC_STATUS = "disabled"
CTC_REPORT: dict = {}
CTC_DECODES: list[dict] = []


def ensure_kraken_env(cfg: Config = None) -> tuple[Path, Path]:
    """Create (once) an isolated venv with kraken; return (python, ketos)."""
    cfg = cfg or CFG
    venv = WORK / cfg.ctc_venv
    py, ketos = venv_bin(venv, "python"), venv_bin(venv, "ketos")
    probe = [str(py), "-c", "import kraken.train.ppocr, kraken.tasks; import kraken; "
                            "print(kraken.__version__ if hasattr(kraken,'__version__') else '')"]
    if py.exists() and subprocess.run(probe, capture_output=True).returncode == 0:
        return py, ketos
    log = CTC_LOGS / "install.log"
    if not py.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    pip_ = [str(py), "-m", "pip", "install", "-q", "--no-cache-dir"]
    run_logged(pip_ + ["--upgrade", "pip"], log)
    if cfg.ctc_torch_index:
        rc = run_logged(pip_ + ["--index-url", cfg.ctc_torch_index,
                                "torch>=2.9,<=2.14", "torchvision"], log)
        if rc != 0:
            print(f"[ctc] torch from {cfg.ctc_torch_index} failed; trying PyPI")
    if run_logged(pip_ + [f"kraken=={cfg.ctc_kraken_version}"], log) != 0:
        raise RuntimeError(f"kraken install failed - see {log}")
    if subprocess.run(probe, capture_output=True).returncode != 0:
        raise RuntimeError(f"kraken imports fail in {venv} - see {log}")
    return py, ketos


def ctc_decode(py: Path, model: Path, ids: Sequence[str], tag: str,
               cfg: Config = None) -> pd.DataFrame:
    """Decode `ids` with one CTC model across the photometric TTA views."""
    cfg = cfg or CFG
    frames = []
    for mode in cfg.ctc_tta_modes:
        line_dir = CTC_LINES if mode == "identity" else CTC_ROOT / f"lines_{mode}"
        paths = write_ctc_lines(ids, None, line_dir, cfg, mode=mode)
        job = CTC_ROOT / "jobs" / f"{tag}_{mode}.json"
        res = CTC_ROOT / "jobs" / f"{tag}_{mode}.out.json"
        job.parent.mkdir(parents=True, exist_ok=True)
        job.write_text(json.dumps({
            "model": str(model), "batch": int(cfg.ctc_infer_batch), "out": str(res),
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "items": [{"id": i, "path": str(p.resolve())} for i, p in zip(ids, paths)],
        }), encoding="utf-8")
        worker = CTC_ROOT / "ctc_worker.py"
        worker.write_text(CTC_WORKER, encoding="utf-8")
        rc = run_logged([str(py), str(worker), str(job)], CTC_LOGS / f"decode_{tag}.log")
        if rc != 0 or not res.exists():
            raise RuntimeError(f"ctc decode {tag}/{mode} failed (rc={rc})")
        df = pd.DataFrame(json.loads(res.read_text(encoding="utf-8")))
        df["view"] = f"ctc:{mode}"
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["text"] = out["text"].fillna("").map(normalize_text)
    return out


def ctc_top1_score(df: pd.DataFrame, ids: Sequence[str]) -> dict:
    ident = df[df.view == "ctc:identity"].set_index("ID")["text"]
    refs = train_df.set_index("ID").loc[list(ids), "Target"].tolist()
    return competition_score(refs, [ident.get(i, "") for i in ids])


def run_ctc_branch(cfg: Config = None) -> list:
    cfg = cfg or CFG
    folds = ctc_folds(cfg)
    stage(f"kraken CTC branch: {len(folds)} fold(s)")
    for d in (CTC_ROOT, CTC_MODELS, CTC_LOGS, CTC_ROOT / "preds"):
        d.mkdir(parents=True, exist_ok=True)

    if cfg.hf_resume and HUB.enabled:
        HUB.pull(["ctc/models/**", "ctc/preds/**"], OUT)

    _env: dict = {}

    def env() -> tuple[Path, Path]:
        """Build the kraken venv only if something actually needs to run in it."""
        if not _env:
            _env["py"], _env["ketos"] = ensure_kraken_env(cfg)
        return _env["py"], _env["ketos"]

    from huggingface_hub import hf_hub_download

    base = Path(hf_hub_download(cfg.ctc_model_id, cfg.ctc_model_file,
                                local_dir=str(CTC_ROOT / "base")))
    print(f"[ctc] base model {cfg.ctc_model_id} -> {base.name} ({CTC_LICENCE})")

    all_paths = write_ctc_lines(train_df.ID.tolist(), train_df.Target.tolist(),
                                CTC_LINES, cfg)
    widths = ctc_width_report(all_paths, cfg)
    print(f"[ctc] {widths['n']} train lines at h={cfg.ctc_line_height}: "
          f"max width {widths['max']}px, p99 {widths['p99']:.0f}px "
          f"-> ketos --max-width {widths['max_width']}")
    if widths["over_limit"]:
        print(f"!! {widths['over_limit']} lines exceed ctc_max_width="
              f"{widths['max_width']} and will be DROPPED by ketos - set "
              f"CFG.ctc_max_width = 0 to size it from the data")
    CTC_REPORT["widths"] = widths

    # Zero-shot check: does the leaderboard result carry over to secretary hand?
    zs_ids = OOF_IDS if OOF_IDS else train_df.ID.tolist()
    zs_path = CTC_ROOT / "preds" / "oof_ctc_zeroshot.csv"
    if zs_path.exists():
        zs = pd.read_csv(zs_path, keep_default_na=False)
    else:
        zs = ctc_decode(env()[0], base, zs_ids, "zeroshot",
                        dataclasses.replace(cfg, ctc_tta_modes=("identity",)))
        zs.to_csv(zs_path, index=False)
        HUB.push_file(zs_path, f"ctc/preds/{zs_path.name}", "ctc zero-shot oof")
    CTC_REPORT["zero_shot"] = ctc_top1_score(zs, zs_ids)
    print(f"[ctc] zero-shot PP-OCRv6 on our lines: "
          f"score {CTC_REPORT['zero_shot']['score']:.5f} "
          f"(CER {CTC_REPORT['zero_shot']['cer']:.4f}, "
          f"WER {CTC_REPORT['zero_shot']['wer']:.4f}) | VLM raw top-1 "
          f"{BASELINE['score']:.5f}")

    decodes = []
    for fold in progress(folds, desc="ctc folds", unit="fold"):
        tag = f"ctc_fold{fold}"
        oof_path = CTC_ROOT / "preds" / f"oof_{tag}.csv"
        test_path = CTC_ROOT / "preds" / f"test_{tag}.csv"
        tr_df, va_df = fold_frames(fold)
        out_dir = CTC_MODELS / tag
        weights = best_ctc_weights(out_dir)

        if cfg.ctc_finetune and weights is None:
            t0 = time.time()
            tr_m = write_manifest([CTC_LINES / f"{i}.png" for i in tr_df.ID],
                                  CTC_ROOT / "manifests" / f"{tag}_train.txt")
            va_m = write_manifest([CTC_LINES / f"{i}.png" for i in va_df.ID],
                                  CTC_ROOT / "manifests" / f"{tag}_val.txt")
            out_dir.mkdir(parents=True, exist_ok=True)
            (CTC_LOGS / f"train_{tag}.log").unlink(missing_ok=True)   # count this run only
            rc = run_logged(ketos_train_cmd(env()[1], base, tr_m, va_m, out_dir, cfg,
                                            max_width=widths["max_width"]),
                            CTC_LOGS / f"train_{tag}.log")
            weights = best_ctc_weights(out_dir)
            _log = (CTC_LOGS / f"train_{tag}.log").read_text(encoding="utf-8", errors="replace")
            _dropped = _log.count("Failed. Replacing")
            if _dropped:
                print(f"!! {tag}: ketos replaced {_dropped} unusable samples (too wide "
                      f"or too many characters for the width) - see the train log")
            CTC_REPORT[f"{tag}_replaced_samples"] = _dropped
            if rc != 0 or weights is None:
                raise RuntimeError(f"ketos train {tag} failed (rc={rc}); "
                                   f"see {CTC_LOGS / f'train_{tag}.log'}")
            print(f"[ctc] {tag}: {weights.name} in {fmt_hms(time.time() - t0)}")
            if cfg.hf_push_checkpoints:
                HUB.push_file(weights, f"ctc/models/{tag}/{weights.name}",
                              f"{tag}: {weights.stem}")
        weights = weights or base

        if oof_path.exists() and test_path.exists():
            oof = pd.read_csv(oof_path, keep_default_na=False)
            test = pd.read_csv(test_path, keep_default_na=False)
        else:
            oof = ctc_decode(env()[0], weights, va_df.ID.tolist(), f"{tag}_oof", cfg)
            test = ctc_decode(env()[0], weights, test_df.ID.tolist(), f"{tag}_test", cfg)
            oof.to_csv(oof_path, index=False)
            test.to_csv(test_path, index=False)
            HUB.push_file(oof_path, f"ctc/preds/{oof_path.name}", f"{tag} oof")
            HUB.push_file(test_path, f"ctc/preds/{test_path.name}", f"{tag} test")
        res = ctc_top1_score(oof, va_df.ID.tolist())
        CTC_REPORT[tag] = res
        print(f"[ctc] {tag}: fine-tuned top-1 OOF score {res['score']:.5f} "
              f"(CER {res['cer']:.4f}, WER {res['wer']:.4f})")
        decodes.append({"tag": tag, "oof": ctc_frame_to_nbest(oof, tag),
                        "test": ctc_frame_to_nbest(test, tag)})
    return decodes


if CFG.ctc_branch:
    try:
        CTC_DECODES = run_ctc_branch()
        CTC_STATUS = f"ok ({len(CTC_DECODES)} fold models)"
    except Exception as exc:
        import traceback

        CTC_DECODES = []
        CTC_STATUS = f"failed: {type(exc).__name__}: {exc}"
        print(f"!! CTC branch {CTC_STATUS}\n{traceback.format_exc()[-2000:]}")
        print("!! continuing VLM-only; the submission is unaffected")
        NOTIFY.send("CTC branch failed - continuing VLM-only",
                    f"{CTC_STATUS}\n\n{traceback.format_exc()[-6000:]}")
print(f"[ctc] status: {CTC_STATUS}")

# Pool the CTC hypotheses with the VLM ones. OOF only for lines the VLM also covers,
# so both families are tuned on exactly the same rows.
_oof_set = set(OOF_IDS)
for _d in CTC_DECODES:
    OOF_NBEST = merge_nbest(OOF_NBEST, _d["oof"], only=_oof_set)
    TEST_NBEST = merge_nbest(TEST_NBEST, _d["test"])
HAS_CTC = any(c.get("family") == "ctc" for i in OOF_IDS for c in OOF_NBEST.get(i, []))

if HAS_CTC:
    _ctc_all = pd.concat([pd.read_csv(CTC_ROOT / "preds" / f"oof_{d['tag']}.csv",
                                      keep_default_na=False) for d in CTC_DECODES])
    CTC_REPORT["oof_top1"] = ctc_top1_score(_ctc_all, OOF_IDS)
    _oracle_ctc = []
    for image_id, ref in zip(OOF_IDS, OOF_REFS):
        cands = {c["text"] for c in OOF_NBEST[image_id]}
        _oracle_ctc.append(max((competition_score([ref], [t])["score"], t) for t in cands)[1])
    CTC_REPORT["oracle_with_ctc"] = competition_score(OOF_REFS, _oracle_ctc)["score"]
    print(f"[ctc] CTC top-1 on the VLM's OOF rows : {CTC_REPORT['oof_top1']['score']:.5f}")
    print(f"[ctc] VLM raw top-1 on the same rows  : {BASELINE['score']:.5f}")
    print(f"[ctc] oracle over VLM+CTC hypotheses  : {CTC_REPORT['oracle_with_ctc']:.5f}")
