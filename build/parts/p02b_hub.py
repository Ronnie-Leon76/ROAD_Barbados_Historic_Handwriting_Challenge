# %% [markdown]
# ### 2.3 Hugging Face Hub sync
#
# A competition run is worth nothing if the pod is destroyed before the weights are
# safe. Set `CFG.hf_repo_id` (and export a write token as `$HF_TOKEN`) and every
# artefact needed to reproduce the leaderboard score is mirrored to a private repo
# as it is produced, not in one push at the end:
#
# | Uploaded | When | Why it is needed to replay the run |
# |---|---|---|
# | `checkpoints/<tag>/` | as each fold finishes | the adapter (or full model) that produced the score |
# | `val_history_<tag>.csv` | as each fold finishes | proves which step was selected and why |
# | `run_fingerprint.json` | at start-up | input digests + library versions the score is conditional on |
# | `post_config.json` | after tuning | the fusion weights, which move the score by more than a training epoch |
# | `submission.csv`, `run_manifest.json` | at the end | the exact rows that were scored |
#
# Going the other way, `CFG.hf_resume = True` pulls the checkpoints back down and
# skips training entirely, which turns a four-hour run into a decoding-only pass —
# useful for re-tuning post-processing without paying for the GPU hours again.
#
# Nothing here is required: with `hf_repo_id` empty every call is a no-op and the
# notebook behaves exactly as before.

# %%
class HubSync:
    """Thin, failure-tolerant wrapper around huggingface_hub uploads.

    Deliberately never raises: a hub outage or an expired token must not destroy a
    training run that has already cost GPU hours. Failures print and continue.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.repo_id = (cfg.hf_repo_id or "").strip()
        self.api = None
        self.enabled = False
        self.errors: list[str] = []
        if not self.repo_id:
            print("[hub] disabled (CFG.hf_repo_id is empty)")
            return
        token = os.environ.get(cfg.hf_token_env) or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        try:
            from huggingface_hub import HfApi

            self.api = HfApi(token=token)
            who = self.api.whoami()["name"]
            self.api.create_repo(self.repo_id, private=cfg.hf_private,
                                 exist_ok=True, repo_type="model")
            self.enabled = True
            print(f"[hub] {who} -> {self.repo_id} "
                  f"({'private' if cfg.hf_private else 'public'})")
        except Exception as exc:
            print(f"[hub] disabled: {type(exc).__name__}: {exc}")
            print(f"[hub] set a write token with  os.environ['{cfg.hf_token_env}'] = '...'")

    def _guard(self, what: str, fn, *args, **kwargs):
        if not self.enabled:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            msg = f"{what}: {type(exc).__name__}: {exc}"
            self.errors.append(msg)
            print(f"[hub] FAILED {msg}")
            return None

    def push_folder(self, local: Path, path_in_repo: str, message: str = "") -> bool:
        local = Path(local)
        if not self.enabled or not local.exists():
            return False
        size = sum(f.stat().st_size for f in local.rglob("*") if f.is_file()) / 1e6
        print(f"[hub] uploading {path_in_repo} ({size:.0f} MB) ...")
        return self._guard(f"upload {path_in_repo}", self.api.upload_folder,
                           folder_path=str(local), path_in_repo=path_in_repo,
                           repo_id=self.repo_id, repo_type="model",
                           commit_message=message or f"add {path_in_repo}") is not None

    def push_file(self, local: Path, path_in_repo: str | None = None,
                  message: str = "") -> bool:
        local = Path(local)
        if not self.enabled or not local.exists():
            return False
        return self._guard(f"upload {local.name}", self.api.upload_file,
                           path_or_fileobj=str(local),
                           path_in_repo=path_in_repo or local.name,
                           repo_id=self.repo_id, repo_type="model",
                           commit_message=message or f"add {local.name}") is not None

    def pull(self, patterns: list[str], dest: Path) -> Path | None:
        """Download matching files from the repo into `dest`. Returns the local root."""
        if not self.enabled:
            return None
        from huggingface_hub import snapshot_download

        return self._guard("snapshot_download", snapshot_download,
                           repo_id=self.repo_id, repo_type="model",
                           allow_patterns=patterns, local_dir=str(dest))

    def write_card(self, body: str) -> None:
        if not self.enabled:
            return
        card = OUT / "README_hub.md"
        card.write_text(body, encoding="utf-8")
        self.push_file(card, "README.md", "update model card")


def model_card(cfg: Config, licence: str, fingerprint: dict,
               runs: list | None = None, scores: dict | None = None) -> str:
    """Model card carrying the numbers and the exact recipe that produced them."""
    base = cfg.model_id
    rows = ""
    for r in runs or []:
        best = r.get("best", {})
        rows += (f"| `{r.get('tag')}` | {best.get('score', float('nan')):.5f} | "
                 f"{best.get('source', '?')} | {best.get('step', '?')} |\n")
    score_lines = "".join(f"- **{k}**: {v}\n" for k, v in (scores or {}).items())
    return f"""---
license: {licence}
base_model: {base}
tags:
  - handwritten-text-recognition
  - ocr
  - historical-documents
  - {"lora" if cfg.finetune_mode == "lora" else "full-finetune"}
library_name: peft
---

# R.O.A.D. Barbados Historic Handwriting - {cfg.finetune_mode.upper()} fine-tune

{"LoRA adapters" if cfg.finetune_mode == "lora" else "Full fine-tune"} of
[`{base}`]({f"https://huggingface.co/{base}"}) for transcribing 17th-19th century
Barbadian deed, will and inventory line crops.

## Results

{score_lines or "_scores are written here when the run completes_"}

| fold | val score | selected from | step |
|---|---|---|---|
{rows or "| _pending_ | | | |"}

## Reproducing this exactly

Run `ROAD_Barbados_HTR_SOTA.ipynb` with the configuration below. The run is
deterministic given the same inputs and device: sampling is seeded per
`(seed, epoch, index)` and cuDNN/cuBLAS determinism is
{"enabled" if cfg.deterministic else "**disabled** for this run"}.

- run id: `{fingerprint.get('run_id')}`
- config hash: `{fingerprint.get('config_hash')}`
- input digests: `{fingerprint.get('input_digests')}`
- seed: `{cfg.seed}`
- device: `{fingerprint.get('device')}` x{fingerprint.get('n_gpu')}, CUDA `{fingerprint.get('cuda')}`
- key versions: `{ {k: v for k, v in fingerprint.get('packages', {}).items() if k in ('torch', 'transformers', 'peft', 'trl')} }`

```python
CFG.finetune_mode = "{cfg.finetune_mode}"
CFG.model_id      = "{base}"
CFG.seed          = {cfg.seed}
CFG.epochs        = {cfg.epochs}
CFG.lr            = {cfg.lr:g}
CFG.lora_r        = {cfg.lora_r}
CFG.lora_alpha    = {cfg.lora_alpha}
CFG.line_height   = {cfg.line_height}
CFG.stackmix      = {cfg.stackmix}
CFG.run_grpo      = {cfg.run_grpo}
```

## Data and licence

Trained only on the challenge's own `Train.csv` images, which are licensed for use
within the competition and are **not** redistributed here. This repository contains
weights and run metadata only.
"""


HUB = HubSync(CFG)
HUB.push_file(OUT / "run_fingerprint.json", "run_fingerprint.json", "record run fingerprint")
