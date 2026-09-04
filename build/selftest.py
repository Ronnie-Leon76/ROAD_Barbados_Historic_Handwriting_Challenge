"""Execute the GPU-free parts of the notebook against the real data.

This is the local CI for the notebook: it runs the data, metric, split, image,
StackMix and fusion sections end-to-end on CPU, then exercises the post-processing
stack on a synthetic N-best list. Anything that needs torch/transformers is skipped.

    python build/selftest.py [--fast]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
PARTS = Path(__file__).resolve().parent / "parts"

CELL_MARK = "# %%"


def cells_of(path: Path):
    """Yield code-cell sources from a percent-format part."""
    text = path.read_text(encoding="utf-8")
    chunks, current, is_code = [], [], False
    for line in text.split("\n"):
        if line.startswith(CELL_MARK):
            if is_code and current:
                chunks.append("\n".join(current))
            current = []
            is_code = not line[len(CELL_MARK):].strip().startswith("[markdown]")
        else:
            current.append(line)
    if is_code and current:
        chunks.append("\n".join(current))
    return chunks


def run_part(ns: dict, name: str, skip_cells: Sequence[int] = ()) -> None:
    path = PARTS / name
    for i, src in enumerate(cells_of(path)):
        if i in skip_cells:
            print(f"    (skipped cell {i})")
            continue
        exec(compile(src, f"{name}#cell{i}", "exec"), ns)


def notebook_pins() -> set:
    """The package specs the notebook's Section 1 install cell passes to pip."""
    import ast

    tree = ast.parse((PARTS / "p01_intro.py").read_text(encoding="utf-8"))
    specs = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "pip"):
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                spec = arg.value
                if spec.startswith("-") or spec == "pip":
                    continue        # flags and `pip install --upgrade pip`
                specs.add(spec)
    return specs


def requirements_pins() -> set:
    lines = (ROOT / "requirements.txt").read_text(encoding="utf-8").split("\n")
    return {ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")}


def check_requirements() -> None:
    """requirements.txt and the notebook install cell must not drift apart."""
    nb = notebook_pins()
    req = requirements_pins()
    # flash-attn is deliberately not installed by `pip install -r`: it needs
    # --no-build-isolation, and the model falls back to SDPA without it.
    optional = {s for s in nb if s.startswith("flash-attn")}
    assert optional, "the notebook no longer installs flash-attn; update this check"
    core = nb - optional
    assert not (core - req), f"missing from requirements.txt: {sorted(core - req)}"
    assert not (req - core), f"in requirements.txt but not installed by the notebook: {sorted(req - core)}"
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "--no-build-isolation" in text and "--no-cache-dir" in text, \
        "requirements.txt should document how to install flash-attn on a split-mount pod"
    print(f"    requirements.txt matches the notebook ({len(core)} pins + flash-attn)")


def prelude(fast: bool) -> dict:
    """Stand-in for the torch-dependent bootstrap cells."""
    ns: dict = {}
    exec("from __future__ import annotations", ns)
    for mod in ("json", "math", "os", "random", "re", "shutil", "sys", "time",
                "warnings", "zipfile", "unicodedata"):
        ns[mod] = __import__(mod)
    ns.update({
        "np": np, "pd": pd, "Path": Path, "Counter": Counter,
        "defaultdict": defaultdict, "dataclass": dataclass, "field": field,
        "asdict": asdict, "Any": Any, "Callable": Callable, "Iterable": Iterable,
        "Sequence": Sequence, "GPU_VRAM_GB": 48.0, "GPU_NAME": "selftest",
        "N_GPU": 1, "DEVICE": "cpu",
    })

    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    ns["set_seed"] = set_seed          # replaced by the real one when p02a runs
    ns["free_gpu"] = lambda: None
    ns["vram_report"] = lambda tag="": None
    ns["FAST"] = fast

    # Minimal torch stand-in so the dataset class and the seeding helpers can be
    # defined without CUDA.
    try:
        import torch  # noqa: F401

        ns["torch"] = torch
    except ImportError:
        import types

        torch_stub = types.ModuleType("torch")
        utils = types.ModuleType("torch.utils")
        data = types.ModuleType("torch.utils.data")

        class _Dataset:
            pass

        data.Dataset = _Dataset
        utils.data = data
        torch_stub.utils = utils
        torch_stub.__version__ = "stub"
        torch_stub.manual_seed = lambda *a, **k: None
        torch_stub.use_deterministic_algorithms = lambda *a, **k: None
        cuda = types.ModuleType("torch.cuda")
        cuda.is_available = lambda: False
        cuda.manual_seed_all = lambda *a, **k: None
        torch_stub.cuda = cuda
        torch_stub.backends = types.SimpleNamespace(
            cudnn=types.SimpleNamespace(benchmark=False, deterministic=False))
        torch_stub.version = types.SimpleNamespace(cuda=None)
        ns["torch"] = torch_stub
    return ns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="smaller StackMix scan")
    args = ap.parse_args()

    os.chdir(ROOT)
    ns = prelude(args.fast)

    print("=" * 70, "\n[1/8] config", sep="")
    check_requirements()
    # Config cell defines Config/CFG/paths; the licence guard needs the network.
    cfg_cells = cells_of(PARTS / "p02_config.py")
    exec(compile(cfg_cells[0], "p02#0", "exec"), ns)
    exec(compile(cfg_cells[1].split("CFG_LICENCE =")[0], "p02#1", "exec"), ns)
    ns["CFG_LICENCE"] = "apache-2.0"
    licence_ok = True
    try:
        ns["check_licence"]("Qwen/Qwen2.5-VL-3B-Instruct")
        licence_ok = False
    except RuntimeError:
        print("    licence guard correctly blocked the research-licensed model")
    assert licence_ok, "licence guard failed to block a non-commercial model"

    # Full fine-tune derivation must refuse the one combination that cannot work.
    _cfg4bit = ns["Config"](finetune_mode="full", load_in_4bit=True)
    try:
        ns["apply_finetune_mode"](_cfg4bit, 48.0)
        raise AssertionError("full fine-tune accepted 4-bit weights")
    except RuntimeError:
        print("    full fine-tune correctly refused 4-bit weights")
    _cfgfull = ns["apply_finetune_mode"](ns["Config"](finetune_mode="full"), 80.0)
    assert _cfgfull.lr == _cfgfull.full_lr and _cfgfull.ema_decay == 0.0, \
        "full fine-tune did not re-derive lr / EMA"
    assert ns["apply_finetune_mode"](ns["Config"](), 48.0).lr == ns["Config"]().lr, \
        "lora mode must not be rewritten"

    print("=" * 70, "\n[1b/8] reproducibility + hub", sep="")
    run_part(ns, "p02a_repro.py")
    fp1 = ns["RUN_FINGERPRINT"]
    fp2 = ns["build_fingerprint"](ns["CFG"])
    assert fp1["run_id"] == fp2["run_id"], "fingerprint is not stable across calls"
    _other = ns["Config"](seed=ns["CFG"].seed + 1, work_dir=ns["CFG"].work_dir)
    assert ns["build_fingerprint"](_other)["run_id"] != fp1["run_id"], \
        "fingerprint ignores the seed"
    _renamed = ns["Config"](hf_repo_id="someone/elsewhere", work_dir=ns["CFG"].work_dir)
    assert ns["build_fingerprint"](_renamed)["config_hash"] == fp1["config_hash"], \
        "fingerprint should ignore hub settings"
    print(f"    fingerprint stable: run_id {fp1['run_id']}")

    run_part(ns, "p02b_hub.py")
    assert ns["HUB"].enabled is False, "hub should be disabled without a repo id"
    assert ns["HUB"].push_file(ROOT / "README.md", "README.md") is False, \
        "disabled hub must no-op instead of raising"
    assert ns["HUB"].push_folder(ROOT / "build", "build") is False
    _card = ns["model_card"](ns["CFG"], "apache-2.0", fp1, [], {"OOF": "0.9"})
    assert "base_model:" in _card and fp1["run_id"] in _card, "model card missing metadata"
    print("    hub no-op path and model card ok")

    print("=" * 70, "\n[1c/8] notifications + tensorboard", sep="")
    # Never send real mail from the selftest: blank the credentials first.
    for _var in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD", "NOTIFY_EMAIL_TO"):
        os.environ[_var] = ""
    run_part(ns, "p02c_notify.py")
    assert ns["NOTIFY"].enabled is False, "notifier must stay off without credentials"
    assert ns["NOTIFY"].send("dry run", "body") is False, "disabled notifier must no-op"
    ns["NOTIFY"].run_started()
    ns["NOTIFY"].checkpoint("fold0", {"score": 0.9, "source": "ema", "step": 100,
                                      "path": "x"}, 12.3, False)
    ns["NOTIFY"].grpo(0, {"sft": 0.9, "grpo": 0.91, "delta": 0.01}, True)
    ns["NOTIFY"].finished(0.92, 0.91, Path("submission.csv"))
    ns["NOTIFY"].failed("unit test", "traceback would go here")
    print("    notifier event formatting ok (nothing sent)")

    run_part(ns, "p02d_tb.py")
    _tb = ns["TBLogger"]("selftest", enabled=True)
    _tb.scalar("val/score", 0.9, 1)
    _tb.scalars({"cer": 0.1, "wer": 0.2, "ignored": "str"}, 1, prefix="val/")
    _tb.text("config", "hello")
    _tb.close()
    _tb_off = ns["TBLogger"]("selftest_off", enabled=False)
    _tb_off.scalar("x", 1.0, 0)   # must not raise with no writer
    _tb_off.close()
    ns["push_tensorboard"]("selftest")   # hub disabled: must be a no-op
    print(f"    tensorboard writer ok ({'events written' if _tb.writer is not None else 'no tensorboard installed, degraded cleanly'})")

    run_part(ns, "p02e_progress.py")
    _bar = ns["progress"](range(3), desc="selftest")
    assert list(_bar) == [0, 1, 2], "progress() must pass the iterable through"
    ns["stage"]("selftest stage A")
    ns["stage"]("selftest stage B")
    assert len(ns["STAGES"]) == 2 and ns["STAGES"][0]["seconds"] is not None, \
        "stage() must close the previous stage"
    assert ns["fmt_hms"](3671) == "1h01m11s", "fmt_hms formatting drifted"
    ns["stage_summary"]()
    ns["STAGES"].clear()
    print("    progress bars and stage timing ok")

    if args.fast:
        ns["CFG"].n_folds = 5

    print("=" * 70, "\n[2/8] data + labels + EDA", sep="")
    run_part(ns, "p03_data.py")

    print("=" * 70, "\n[3/8] metric", sep="")
    run_part(ns, "p04_metric.py")

    print("=" * 70, "\n[4/8] split", sep="")
    run_part(ns, "p05_split.py")

    print("=" * 70, "\n[5/8] image pipeline", sep="")
    if args.fast:
        # Warm the cache on a subset only.
        src = (PARTS / "p06_images.py").read_text(encoding="utf-8")
        src = src.replace("warm_cache(list(train_df.ID) + list(test_df.ID))",
                          "warm_cache(list(train_df.ID[:250]) + list(test_df.ID[:50]))")
        src = src.replace("train_df.ID.sample(400, random_state=0)",
                          "train_df.ID[:250].sample(200, random_state=0)")
        tmp = PARTS / "_tmp_p06.py"
        tmp.write_text(src, encoding="utf-8")
        try:
            run_part(ns, "_tmp_p06.py")
        finally:
            tmp.unlink(missing_ok=True)
    else:
        run_part(ns, "p06_images.py")

    print("=" * 70, "\n[6/8] stackmix", sep="")
    src = (PARTS / "p07_stackmix.py").read_text(encoding="utf-8")
    if args.fast:
        src = src.replace("_bank, _donors = build_word_bank(train_df, CFG)",
                          "_bank, _donors = build_word_bank(train_df.head(250), CFG)")
    tmp = PARTS / "_tmp_p07.py"
    tmp.write_text(src, encoding="utf-8")
    try:
        run_part(ns, "_tmp_p07.py")
    finally:
        tmp.unlink(missing_ok=True)

    print("=" * 70, "\n[7/8] dataset + collator (numpy path only)", sep="")
    ds_src = cells_of(PARTS / "p08_dataset.py")[0]
    head, _, _ = ds_src.partition("class OCRCollator")
    exec(compile(head, "p08#head", "exec"), ns)
    ns["Image"] = __import__("PIL.Image", fromlist=["Image"])
    sample_ds = ns["LineDataset"](ns["train_df"].head(8), ns["CFG"], train=True,
                                  renderer=ns.get("STACKMIX"), seed=0)
    item = sample_ds[0]
    assert item["image"].mode == "RGB" and item["text"], item
    print(f"    LineDataset ok: {len(sample_ds)} items "
          f"(real {sample_ds.n_real} + synthetic {sample_ds.n_synth}), "
          f"first image {item['image'].size}")
    if sample_ds.n_synth:
        syn = sample_ds[sample_ds.n_real]
        print(f"    synthetic sample: {syn['image'].size} | {syn['text'][:60]!r}")
    assert ns["find_last_subsequence"]([1, 2, 3, 9, 1, 2, 3], [1, 2, 3]) == 7
    assert ns["find_last_subsequence"]([1, 2], [3]) == -1
    print("    find_last_subsequence ok")

    print("=" * 70, "\n[8/8] fusion + post-processing", sep="")
    run_part(ns, "p14_fusion.py")
    exercise_postprocessing(ns)

    print("=" * 70)
    print("SELFTEST PASSED")
    return 0


def exercise_postprocessing(ns: dict) -> None:
    """Drive build_features / apply_post on a synthetic N-best list."""
    src = (PARTS / "p15_tune.py").read_text(encoding="utf-8")
    head = src.split("FOLD_OF = dict(")[0]
    for chunk in head.split("\n# %%"):
        exec(compile(chunk, "p15#head", "exec"), ns)

    train_df = ns["train_df"]
    rng = np.random.default_rng(0)
    ids = train_df.ID.tolist()[:400]
    refs = train_df.set_index("ID").loc[ids, "Target"].tolist()

    def corrupt(text: str, rate: float) -> str:
        chars = list(text)
        for i in range(len(chars)):
            if rng.random() < rate:
                chars[i] = chr(int(rng.integers(97, 123)))
        return "".join(chars)

    nbest = {}
    for image_id, ref in zip(ids, refs):
        cands = []
        for view in range(4):
            text = corrupt(ref, 0.05)
            cands.append({"text": text, "score": -float(rng.uniform(0.02, 0.3)),
                          "view": f"v{view}", "rank": 0, "model": "m0"})
        cands.append({"text": ref, "score": -0.4, "rank": 1, "view": "v0", "model": "m0"})
        nbest[image_id] = cands

    fold_of = dict(zip(train_df.ID, train_df.fold))
    feats = ns["build_features"](nbest, ids, fold_of, restrict=True, collapse=True)
    pc_plain = ns["PostConfig"](use_rover=False, repair=False)
    pc_rover = ns["PostConfig"](use_rover=True, rover_level="char", repair=False)

    plain = ns["apply_post"](feats, ids, pc_plain)
    fused = ns["apply_post"](feats, ids, pc_rover)
    s_plain = ns["competition_score"](refs, [plain[i] for i in ids])["score"]
    s_fused = ns["competition_score"](refs, [fused[i] for i in ids])["score"]
    s_raw = ns["competition_score"](refs, [nbest[i][0]["text"] for i in ids])["score"]
    print(f"    synthetic N-best: raw {s_raw:.4f} | rescored {s_plain:.4f} | ROVER {s_fused:.4f}")
    assert s_fused > s_raw + 0.01, "ROVER failed to fix independent random errors"

    lm, lex = ns["fold_resources"](0)
    good = lm.logprob_per_char("and delivered in the presence of")
    junk = lm.logprob_per_char("qzx wvkj bbbb zzzz")
    print(f"    charLM: real text {good:.3f} vs junk {junk:.3f} logprob/char")
    assert good > junk, "character LM does not separate real text from noise"

    repaired = ns["repair_words"]("Sealed and deliverd in the presance of", lex,
                                  min_sim=85.0, min_freq=2)
    print(f"    lexicon repair: {repaired!r}")
    assert repaired == "Sealed and delivered in the presence of", repaired

    # In-vocabulary archaic spellings must survive untouched.
    kept = ns["repair_words"]("Sealed and delivered in the prsence of us", lex,
                              min_sim=85.0, min_freq=2)
    assert kept == "Sealed and delivered in the prsence of us", kept
    print(f"    archaic spelling preserved: {kept!r}")

    assert ns["collapse_repeats"]("of the of the of the of the of the") == "of the of the of the"
    kept_chars = ns["restrict_charset"]("and \u00e9\u00f1\u4e2d the")
    assert kept_chars == "and  the", repr(kept_chars)
    print("    collapse_repeats + charset restriction ok")


if __name__ == "__main__":
    sys.exit(main())
