# %% [markdown]
# ### 2.6 Progress reporting
#
# A full run is a few hours across a dozen stages, and half of them are silent loops
# over thousands of images. Two things make that legible, and both have to work in a
# terminal *and* in a log file that nobody is watching live:
#
# * **`progress(...)`** wraps `tqdm` with one house style. In a terminal it behaves
#   normally. When stdout is not a TTY — which is every background run — it switches
#   to ASCII and refreshes at most every `PROGRESS_INTERVAL` seconds (30 by default)
#   instead of ten times a second. That matters more than it sounds: `tqdm` redraws
#   with a carriage return, so a default-rate bar writes tens of thousands of updates
#   into a log file. At one line every 30 seconds the same bar becomes a readable
#   timeline of the run.
# * **`stage(...)`** prints a banner and records how long each phase took. The
#   summary at the end tells you where the wall clock actually went, which is the
#   information you need to decide what to cut when the deadline is close.
#
# `PROGRESS_INTERVAL` is read from the environment so a background run can be made
# chattier (`PROGRESS_INTERVAL=5`) without editing anything.

# %%
from tqdm.auto import tqdm

INTERACTIVE = bool(getattr(sys.stdout, "isatty", lambda: False)())
PROGRESS_INTERVAL = float(os.environ.get("PROGRESS_INTERVAL", "30"))
RUN_T0 = time.time()


def fmt_hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def progress(iterable=None, *, total=None, desc="", unit="it", leave=True, **kw):
    """tqdm with one house style, tuned for whichever output we actually have."""
    opts = dict(desc=desc, unit=unit, leave=leave, total=total,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}]{postfix}")
    if INTERACTIVE:
        opts.update(ncols=100, mininterval=0.2, smoothing=0.3)
    else:
        # One tidy, timestamped line every PROGRESS_INTERVAL seconds.
        opts.update(ncols=80, ascii=True, mininterval=PROGRESS_INTERVAL,
                    miniters=1, smoothing=0.1, file=sys.stdout)
    opts.update(kw)
    return tqdm(iterable, **opts)


STAGES: list[dict] = []


def stage(name: str) -> None:
    """Close the previous phase, open a new one, and say where we are."""
    now = time.time()
    if STAGES:
        STAGES[-1]["seconds"] = now - STAGES[-1]["t0"]
    STAGES.append({"name": name, "t0": now, "seconds": None})
    print(f"\n{'=' * 78}")
    print(f"[stage {len(STAGES)}] {name}    (+{fmt_hms(now - RUN_T0)} into the run)")
    print("=" * 78)


def stage_summary() -> "pd.DataFrame":
    now = time.time()
    if STAGES and STAGES[-1]["seconds"] is None:
        STAGES[-1]["seconds"] = now - STAGES[-1]["t0"]
    total = max(now - RUN_T0, 1e-6)
    df = pd.DataFrame([{"stage": s["name"],
                        "minutes": round((s["seconds"] or 0) / 60, 1),
                        "share": f"{100 * (s['seconds'] or 0) / total:5.1f}%"}
                       for s in STAGES])
    print(f"\nwall clock {fmt_hms(total)} over {len(STAGES)} stages")
    print(df.to_string(index=False))
    return df


print(f"[progress] {'interactive terminal' if INTERACTIVE else 'headless'}: "
      f"bars refresh every {0.2 if INTERACTIVE else PROGRESS_INTERVAL:g}s")
