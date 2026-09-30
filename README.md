# R.O.A.D. Barbados Historic Handwriting Challenge

Pipeline for transcribing 17th–19th century Barbadian deed, will and inventory line
crops. It ships in two forms, built from the same source cells so they cannot drift:

- [`ROAD_Barbados_HTR_SOTA.ipynb`](ROAD_Barbados_HTR_SOTA.ipynb) — the interactive
  deliverable, with the diagnostics and plots inline.
- [`ROAD_Barbados_Kraken_PP_OCRv6.ipynb`](ROAD_Barbados_Kraken_PP_OCRv6.ipynb) — a
  **separate** Kraken-first notebook (fresh kernel). Fine-tunes the 15.9 M PP-OCRv6
  line model on all five folds and optionally ensembles the Qwen submission.
- `run_pipeline.py` — a headless script for unattended runs, generated with
  `python build/build_script.py`.

## What it does

A LoRA fine-tune of an Apache-2.0 vision-language model on the challenge's own line
images, decoded with beam search plus multi-scale test-time augmentation, and fused
across folds with ROVER voting, character n-gram rescoring and lexicon repair. A
second, much smaller recogniser — kraken's PP-OCRv6 CTC model — is fine-tuned on the
same lines and joins the fusion as an independent voter, kept only if it wins
out-of-fold.

| Stage | Section | Summary |
|---|---|---|
| Rules & licence guard | 0, 2 | Refuses to load a model whose licence bars commercial use |
| Reproducibility | 2.2 | Deterministic kernels, per-sample seeding, run fingerprint over inputs and versions |
| Hub sync | 2.3 | Mirrors checkpoints and artefacts to Hugging Face; `hf_resume` replays a run without retraining |
| Email alerts | 2.4 | Mail on checkpoint upload, GRPO verdict, completion and failure |
| TensorBoard | 2.5 | Loss, learning rate **and** generation CER/WER/score curves, mirrored to the hub |
| Progress | 2.6 | Stage banners, `tqdm` bars and a training heartbeat, throttled so a log file stays readable |
| Data & EDA | 3 | Downloads `images.zip`, indexes 5 472 crops, charset and geometry analysis |
| Metric | 4 | Weighted WER/CER replica of the leaderboard formula, unit-tested |
| Split | 5 | Near-duplicate-aware `StratifiedGroupKFold`, verified leak-free |
| Image pipeline | 6 | Background flattening, projection deskew, patch-grid height normalisation |
| StackMix | 7 | Word crops segmented from `Train.csv` and recombined onto real backgrounds |
| Training | 8–12 | bf16 LoRA **or full fine-tune**, cosine annealing with warm-up and floor, NEFTune, adapter EMA, generation-based checkpoint selection |
| GRPO | 13 | RL stage optimising the competition metric directly, kept only if it beats its parent out-of-fold |
| Decoding | 14 | Batched beam search producing N-best lists with sequence scores |
| CTC member | 14.1 | kraken PP-OCRv6 (15.9M params, Apache-2.0) fine-tuned per fold in an isolated venv; zero-shot and fine-tuned scores reported |
| Fusion | 15 | ROVER + rescoring + lexicon repair, tuned out-of-fold twice (VLM only, VLM + CTC) with the better set kept |
| Submission | 16–18 | `submission.csv` plus a run manifest and a "what to try next" list |

## The three switches that change the shape of a run

```python
CFG.finetune_mode = "lora"      # or "full" - lr, optimiser, batch shape and EMA follow
CFG.run_grpo      = True        # on by default; discarded automatically if it loses
CFG.hf_repo_id    = "you/road-barbados-htr"   # plus $HF_TOKEN, to mirror everything
```

`finetune_mode="full"` unfreezes every weight and re-derives the hyperparameters that
cannot be shared with LoRA — a 100× lower learning rate, 8-bit Adam, a quartered
micro-batch and no EMA (its fp32 shadow would be model-sized). It refuses outright to
run on 4-bit weights, and prints a VRAM estimate against the detected card. On 3 300
lines per fold LoRA is usually the better model, but the two make different mistakes,
so a full-fine-tuned 4B fold is a strong extra member in the ROVER vote.

## The kraken CTC member

The [BHL OCR leaderboard](https://huggingface.co/spaces/finebooks/bhl-ocr-leaderboard)
(2026-09-03) found kraken's 15.9M-parameter PP-OCRv6 port reading 18th–19th-century
book pages on a par with VLMs hundreds of times larger: reading CER .034 [.024–.044]
vs .051 [.029–.077] for Qwen3.5-9B, and no looped pages at all (Qwen: 1.5 %).

That is evidence for adding it, not for swapping the VLM out:

* the board is **zero-shot on printed books**; this challenge is **fine-tuned on
  handwriting**, where the VLM has 3 300 in-domain lines to learn from;
* the reading-CER intervals **overlap**, which the board itself calls noise — the
  differences outside noise are the failure modes (loops, invented text on sparse
  lines), which CTC cannot produce by construction;
* two architectures make different mistakes, which is what ROVER turns into accuracy.

So Section 14.1 fine-tunes it per fold (minutes each), decodes with photometric TTA,
and gives Section 15 two things: extra hypotheses, and a `c_sim` feature measuring how
far each VLM hypothesis is from what the CTC model read. Section 15 tunes the fusion
with and without it and keeps whichever scores higher out-of-fold. The run summary
prints the zero-shot score on *our* lines, the fine-tuned score and the gate verdict,
so whether the leaderboard result transfers is measured rather than assumed.

kraken 7.1 pins `torch>=2.9`, `scipy~=1.15` and `scikit-learn~=1.7`, so it is
installed into its own virtualenv (`.venv_kraken`, ~3 GB on first run) and driven by
subprocess. If anything in that branch fails, the run emails you and continues
VLM-only. Switch it off with `--no-ctc` or `CFG.ctc_branch = False`; set
`CFG.ctc_torch_index` if your driver needs a different CUDA wheel than cu126.

## Kraken-first notebook (`ROAD_Barbados_Kraken_PP_OCRv6.ipynb`)

The Qwen LoRA run on one fold scored **0.90491 OOF / 0.90485 public**. First place
is **0.9399**. GRPO added `+0.00041`. That is a recogniser gap, not a fusion gap.

`ROAD_Barbados_Kraken_PP_OCRv6.ipynb` is a **separate kernel**: do not run it
inside the Qwen environment (kraken 7.1 wants `torch>=2.9`). It fine-tunes
`small-models-for-glam/kraken-ppocrv6-medium` (Apache-2.0, 15.9 M, native line
height 128) on **all five** group-aware folds, optionally the `small` sibling as
a second CTC family, decodes with photometric TTA, and fuses with ROVER + a
fold-safe character LM + lexicon repair. Crops are treated as **one line** — the
starter-kit page segmenter is the wrong tool for this dataset.

If `submission_road_htr.csv` (the Qwen file) is next to the notebook it is blended
in at test time, but only when the two strings are close. A fluent VLM
hallucination is not allowed to overwrite a literal CTC reading.

```bash
python build/build_notebook.py --parts build/kraken_parts --out ROAD_Barbados_Kraken_PP_OCRv6.ipynb
```

Run all cells on a fresh PyTorch 2.9+ image. A 16 M model does not need 40 GB;
a 16 GB card is enough. Output: `submission_kraken.csv` and, when the Qwen file
is present, `submission_ensemble.csv`. Submit both; keep whichever the public
board prefers.

## Running it

1. Rent a GPU with **at least 40 GB** of VRAM (L40S, A100, H100) on a PyTorch ≥ 2.6
   image. The notebook auto-scales down to a 4B backbone with 4-bit QLoRA on smaller
   cards, but the 8B configuration is what the defaults target.
2. Put `Train.csv`, `Test.csv` and `SampleSubmission.csv` next to the notebook.
   `images.zip` is downloaded automatically on first run.
3. Run all cells.  Section 1 installs its own dependencies, so nothing is needed
   beforehand. Outside the notebook — for `run_pipeline.py`, or to build the image
   for a pod — use `pip install -r requirements.txt`. PyTorch is not in that file
   on purpose: the right wheel depends on your CUDA version, and every GPU image
   already ships one. With the default `CFG.train_folds = (0,)` a full pass takes roughly
   two hours end to end.
4. For a competitive submission set `CFG.train_folds = (0, 1, 2, 3, 4)` and re-run —
   ensembling across folds is the single most reliable improvement in the pipeline.

Everything is driven by the `CFG` object in Section 2; no other cell needs editing.

## Running it unattended

```bash
cp .env.example .env          # then fill in HF_TOKEN, SMTP_*, NOTIFY_EMAIL_TO
python build/check_notifications.py          # sends one test email
python build/build_script.py                 # emits run_pipeline.py
./run_background.sh --folds 0 1 2 3 4 --hf-repo you/road-barbados-htr
```

`run_background.sh` (or `run_background.ps1` on Windows) detaches the job with
`nohup`, writes the PID to `logs/run.pid` and returns immediately, so the run
survives the SSH session ending. Everything the script prints is also mirrored to a
timestamped file under `logs/`.

`run_pipeline.py` takes the settings you would otherwise edit by hand:

```bash
python run_pipeline.py --fast                      # 1 fold, 1 epoch: a plumbing check
python run_pipeline.py --mode full --model Qwen/Qwen3-VL-4B-Instruct
python run_pipeline.py --resume-from-hub           # decode a previous run, no training
python run_pipeline.py --set lora_r=64 --set line_height=192
python run_pipeline.py --install                   # first launch on a fresh pod
python run_pipeline.py --progress-interval 5       # chattier progress bars in the log
python run_pipeline.py --no-ctc                    # skip the kraken CTC member
python run_pipeline.py --set ctc_epochs=80         # longer CTC fine-tune
```

`--set FIELD=VALUE` reaches any field on `CFG` and is validated against the dataclass,
so a typo stops the run in the first second rather than the third hour.

## Watching a run

**Email** goes to `NOTIFY_EMAIL_TO` when the run starts, when each fold's checkpoint
finishes uploading (with its validation score and hub URL), on each GRPO accept or
reject verdict with the delta, on completion with the out-of-fold score, and on any
uncaught failure with the traceback. A bootstrap handler covers start-up crashes that
happen before the notifier itself exists — a missing CSV or a bad token would
otherwise only appear in a log nobody is watching. Suppress it all with `--no-email`.

**The log itself** is designed to be read. Every phase opens with a `[stage n]`
banner carrying the elapsed wall clock, every long loop has a `tqdm` bar, and
training prints a heartbeat line with the step count, loss, learning rate and a
projected finish time. Bars behave differently depending on where the output goes: in
a terminal they redraw as usual, and when stdout is a file they switch to ASCII and
refresh once every 30 seconds, so a bar that would otherwise write tens of thousands
of carriage returns into `logs/` becomes a readable timeline instead. Change the
cadence with `--progress-interval 5` or `PROGRESS_INTERVAL=5`. The Trainer's own bar
is disabled in that mode for the same reason. A `stage_times.csv` at the end records
where the wall clock actually went.

```
14:02:11 [stage 6] training 5 fold(s), mode lora    (+22m14s into the run)
14:07:44 [fold0_seed42] step 120/2400 (  5.0%) loss 0.8123 lr 1.18e-04 | 5m31s elapsed, ~1h47m left | epoch 0.25
14:12:03 stackmix scan:  37%|##9     | 1511/4098 [00:30<00:51, 50.2line/s], accepted 150
```

**TensorBoard** gets more than the Trainer's loss curve. Because loss is not the
quantity being optimised, each validation pass also logs `val/score`, `val/cer` and
`val/wer` for both the live weights and the EMA, so you can see whether the cosine
tail is still buying anything and whether EMA is ahead of raw:

```bash
tensorboard --logdir artifacts/runs --port 6006
```

With `hf_repo_id` set, the event files are pushed with each fold and the repo grows a
**Training metrics** tab that renders the same curves in a browser — no pod access
and no port forwarding, which is the point when the job is headless.

Neither channel can take down a run: a refused SMTP connection, a missing
`tensorboard` package or a hub outage prints a warning and training continues.

## Secrets

Credentials live in `.env`, which is gitignored; `.env.example` documents the keys.
Nothing is hard-coded in the notebook or the script, so both can be shared as-is.
Gmail requires a 16-character [App Password](https://myaccount.google.com/apppasswords)
rather than the account password. If a token or app password is ever exposed, revoke
it at the provider — rotating it in `.env` alone is not enough.

## Reproducing a score

Section 2.2 prints a `run_id` derived from the seed, the SHA-256 of each input CSV
and every config field that affects the numbers. It travels with the manifest and the
model card, so a claimed score can be checked rather than trusted. Sampling is seeded
per `(seed, epoch, index)` rather than per stream, which makes augmentation identical
regardless of dataloader worker count; `CFG.deterministic` additionally pins cuDNN and
cuBLAS at a cost of roughly 5–10% throughput. A different GPU architecture or CUDA
build can still shift the last digits — the fingerprint records the device so that
case is visible.

With `CFG.hf_repo_id` set, each fold is uploaded as it finishes, so an interrupted run
loses at most the fold in flight. Setting `CFG.hf_resume = True` on a fresh pod pulls
the checkpoints back and skips training entirely, going straight to decoding — the
same weights, therefore the same score, without repeating the GPU hours.

## Rules compliance

The organisers' discussion-board rulings are enforced rather than assumed:

* **Model licences.** Only commercially licensed weights. `check_licence()` queries
  the Hugging Face API and raises on anything outside the allow-list, which is why
  `Qwen2.5-VL-3B-Instruct` (`qwen-research`) and `churro-3B` are blocked while
  `Qwen3-VL-8B-Instruct` and `Qwen2.5-VL-7B-Instruct` (both Apache-2.0) are allowed.
  The kraken member (`small-models-for-glam/kraken-ppocrv6-medium`) is Apache-2.0 and
  passes through the same guard.
* **StackMix.** Explicitly permitted as training-data augmentation. Only `Train.csv`
  images are segmented, the whole process is code-driven and seeded, and labels are
  the concatenation of the source words.
* **No external data.** The only text resource is the challenge's own training
  transcriptions.
* **Test-time pseudo-labelling is disabled** pending an organiser ruling.

## Repository layout

```
ROAD_Barbados_HTR_SOTA.ipynb   generated notebook — the interactive deliverable
ROAD_Barbados_Kraken_PP_OCRv6.ipynb  Kraken-first notebook (fresh kernel)
run_pipeline.py                generated script — the unattended deliverable
run_background.sh / .ps1       detached launcher
requirements.txt               pins for a non-notebook environment (no torch)
.env / .env.example            secrets (gitignored) and their template
build/parts/p*.py              source cells, jupytext "percent" format, in run order
build/kraken_parts/k*.py       source cells for the Kraken notebook
build/build_notebook.py        parts -> .ipynb
build/build_script.py          parts -> run_pipeline.py
build/selftest.py              CPU test of the GPU-free half of the pipeline
build/check_notifications.py   SMTP preflight
build/proto_segment.py         standalone word-segmentation experiment
.venv_kraken/                  isolated kraken env, created on first run (gitignore it)
artifacts/                     EDA plots, checkpoints, predictions, TensorBoard runs
artifacts_kraken/              Kraken-notebook outputs
logs/                          timestamped output from background runs
```

Both deliverables are generated, so edit `build/parts/*.py` and rebuild:

```bash
python build/build_notebook.py
python build/build_script.py
python build/selftest.py          # ~4 minutes, needs the data present
```

`selftest.py` runs the data, metric, split, image, StackMix and fusion sections
against the real images on CPU, and checks the metric, ROVER, character LM and
lexicon repair against known-answer cases. It also asserts that `requirements.txt`
and the notebook's install cell list exactly the same packages, so the two cannot
drift apart unnoticed.
