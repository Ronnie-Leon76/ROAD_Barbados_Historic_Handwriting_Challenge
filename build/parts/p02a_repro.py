# %% [markdown]
# ### 2.2 Reproducibility contract
#
# "Re-run and get the same number" needs three separate things, and it is worth
# being precise about which one each mechanism buys:
#
# 1. **Same sampling.** Seeding `random`, `numpy` and `torch` fixes the fold split,
#    the augmentation draws and the StackMix composition. This notebook goes further:
#    every augmented sample is drawn from `default_rng((seed, epoch, index))` rather
#    than from a stream, so sample *i* of epoch *e* is identical no matter how many
#    dataloader workers there are or in what order they happen to run.
# 2. **Same arithmetic.** GPU reductions are non-deterministic by default because
#    floating-point addition is not associative and the kernels race. `deterministic`
#    mode pins cuBLAS workspaces and forbids the non-deterministic kernels, which
#    costs roughly 5-10% throughput and is worth paying for a competition run.
# 3. **Same inputs and same code.** A seed is useless if the CSV or the library
#    versions moved underneath it. `RUN_FINGERPRINT` hashes the actual input files
#    and records the versions, and it travels with every checkpoint pushed to the hub.
#
# What still cannot be promised: a different GPU architecture, or a different CUDA
# or cuBLAS build, can produce a slightly different result from bit-identical inputs.
# The fingerprint records the device so that case is at least visible rather than
# mysterious.

# %%
import hashlib
import platform


def set_seed(seed: int, deterministic: bool | None = None) -> None:
    """Seed every RNG this pipeline touches, optionally pinning kernel choice too."""
    if deterministic is None:
        deterministic = getattr(CFG, "deterministic", False)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        # Must be set before the first cuBLAS handle is created to take effect.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:
            print(f"[repro] deterministic algorithms unavailable ({exc})")
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


def file_digest(path: Path, chunk: int = 1 << 20) -> str:
    if not Path(path).exists():
        return "missing"
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


def package_versions() -> dict:
    import importlib

    out = {}
    for name in ("torch", "transformers", "peft", "trl", "accelerate", "datasets",
                 "bitsandbytes", "numpy", "pandas", "cv2", "rapidfuzz", "sklearn"):
        try:
            out[name] = str(getattr(importlib.import_module(name), "__version__", "?"))
        except Exception:
            out[name] = "absent"
    return out


def build_fingerprint(cfg: Config) -> dict:
    """Everything that has to match for two runs to be comparable."""
    inputs = {name: file_digest(WORK / name)
              for name in (cfg.train_csv, cfg.test_csv, cfg.sample_sub_csv)}
    # Config fields that change the numbers, not the plumbing. Hub and I/O settings
    # are excluded so that pushing to a different repo does not invalidate a match.
    volatile = {"hf_repo_id", "hf_private", "hf_push_checkpoints", "hf_push_submission",
                "hf_resume", "hf_token_env", "work_dir", "data_dir", "out_dir",
                "dataloader_workers", "infer_batch_size",
                "ctc_venv", "ctc_torch_index", "ctc_infer_batch"}
    material = {k: v for k, v in asdict(cfg).items() if k not in volatile}
    config_hash = hashlib.sha256(
        json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()[:16]
    fp = {
        "config_hash": config_hash,
        "input_digests": inputs,
        "packages": package_versions(),
        "device": GPU_NAME,
        "n_gpu": N_GPU,
        "cuda": torch.version.cuda if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "deterministic": bool(cfg.deterministic),
        "seed": cfg.seed,
    }
    fp["run_id"] = hashlib.sha256(
        json.dumps({k: fp[k] for k in ("config_hash", "input_digests", "seed")},
                   sort_keys=True).encode()).hexdigest()[:12]
    return fp


set_seed(CFG.seed, CFG.deterministic)
RUN_FINGERPRINT = build_fingerprint(CFG)
with open(OUT / "run_fingerprint.json", "w") as fh:
    json.dump(RUN_FINGERPRINT, fh, indent=2)

print(f"[repro] run id        {RUN_FINGERPRINT['run_id']}")
print(f"[repro] config hash   {RUN_FINGERPRINT['config_hash']}")
print(f"[repro] input digests {RUN_FINGERPRINT['input_digests']}")
print(f"[repro] determinism   {'pinned' if CFG.deterministic else 'off (faster, not bit-exact)'}")
print("[repro] versions      " + ", ".join(
    f"{k} {v}" for k, v in RUN_FINGERPRINT["packages"].items() if v != "absent"))
