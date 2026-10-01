# %% [markdown]
# ### 2.4 Email notifications
#
# A five-fold run with GRPO is several hours on a rented pod, and the failure modes
# that matter are the silent ones: an OOM at fold three, an expired hub token, a
# reward function that quietly returns the same value for every completion. This
# sends mail at the points where a human would actually want to intervene:
#
# | Event | Sent when | Why you care |
# |---|---|---|
# | `run started` | after the config is resolved | confirms the pod really is training what you think |
# | `checkpoint uploaded` | each fold, after the hub push | the weights are safe; includes the fold's validation score |
# | `grpo verdict` | each GRPO stage | accepted or rejected, with the delta |
# | `run finished` | after the submission is written | final out-of-fold score and where the file is |
# | `run FAILED` | any uncaught exception | full traceback, so you can fix it without opening the pod |
#
# Credentials come from the environment (`SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`,
# `SMTP_PASSWORD`, `NOTIFY_EMAIL_TO`), loaded from a gitignored `.env` by `Section 2`.
# Nothing is hard-coded here, so this notebook can be shared without leaking a
# password. With Gmail the password must be a 16-character **App Password**, not the
# account password.
#
# Like the hub layer, the notifier never raises: a mail server that refuses a
# connection must not kill a training run.

# %%
import smtplib
import traceback
from email.message import EmailMessage


class Notifier:
    """Sends short status emails over SMTP, and never lets a mail failure escalate."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.to = os.environ.get("NOTIFY_EMAIL_TO", "").strip()
        self.host = os.environ.get("SMTP_HOST", "smtp.gmail.com").strip()
        self.port = int(os.environ.get("SMTP_PORT", "587") or 587)
        self.user = os.environ.get("SMTP_USER", "").strip()
        self.password = os.environ.get("SMTP_PASSWORD", "").strip()
        self.enabled = bool(cfg.notify_email and self.to and self.host
                            and self.user and self.password)
        self.sent = 0
        self.failures: list[str] = []
        self.t0 = time.time()
        if self.enabled:
            print(f"[notify] email -> {self.to} via {self.host}:{self.port}")
        elif not cfg.notify_email:
            print("[notify] disabled (CFG.notify_email = False)")
        else:
            missing = [n for n, v in (("SMTP_HOST", self.host), ("SMTP_USER", self.user),
                                      ("SMTP_PASSWORD", self.password),
                                      ("NOTIFY_EMAIL_TO", self.to)) if not v]
            print(f"[notify] disabled (missing {', '.join(missing)} - see .env.example)")

    def _elapsed(self) -> str:
        m, s = divmod(int(time.time() - self.t0), 60)
        h, m = divmod(m, 60)
        return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"

    def send(self, subject: str, body: str, urgent: bool = False) -> bool:
        prefix = "[ROAD-HTR]" + (" FAILURE" if urgent else "")
        line = f"{prefix} {subject}"
        print(f"[notify] {line}")
        if not self.enabled:
            return False
        run = RUN_FINGERPRINT.get("run_id", "?")
        footer = (f"\n\n--\nrun {run} | {GPU_NAME} | elapsed {self._elapsed()}"
                  f"\nmodel {self.cfg.model_id} | mode {self.cfg.finetune_mode}"
                  + (f"\nhub https://huggingface.co/{self.cfg.hf_repo_id}"
                     if self.cfg.hf_repo_id else ""))
        msg = EmailMessage()
        msg["Subject"] = line
        msg["From"] = self.user
        msg["To"] = self.to
        msg.set_content(body + footer)
        try:
            with smtplib.SMTP(self.host, self.port, timeout=30) as server:
                server.starttls()
                server.login(self.user, self.password)
                server.send_message(msg)
            self.sent += 1
            return True
        except Exception as exc:
            note = f"{type(exc).__name__}: {exc}"
            self.failures.append(note)
            print(f"[notify] send failed ({note})")
            return False

    # ------------------------------------------------------------- events --
    def run_started(self) -> None:
        self.send(
            f"run started on {GPU_NAME}",
            f"Training {self.cfg.model_id} ({self.cfg.finetune_mode}) on folds "
            f"{list(self.cfg.train_folds)}.\n"
            f"epochs {self.cfg.epochs:g} | lr {self.cfg.lr:g} | batch {EFFECTIVE_BATCH}\n"
            f"stackmix {self.cfg.stackmix} | grpo {self.cfg.run_grpo} | "
            f"deterministic {self.cfg.deterministic}\n"
            f"run id {RUN_FINGERPRINT['run_id']} (config {RUN_FINGERPRINT['config_hash']})",
        )

    def checkpoint(self, tag: str, best: dict, minutes: float, pushed: bool) -> None:
        where = (f"https://huggingface.co/{self.cfg.hf_repo_id}/tree/main/checkpoints/{tag}"
                 if pushed else f"local only: {best.get('path')}")
        self.send(
            f"checkpoint {tag} - val {best.get('score', float('nan')):.5f}",
            f"Fold finished in {minutes:.1f} min.\n\n"
            f"best score  {best.get('score', float('nan')):.5f}\n"
            f"selected    {best.get('source')} at step {best.get('step')}\n"
            f"uploaded    {where}",
        )

    def grpo(self, fold: int, scores: dict, accepted: bool) -> None:
        self.send(
            f"grpo fold {fold} {'accepted' if accepted else 'rejected'} "
            f"({scores['delta']:+.5f})",
            f"SFT  {scores['sft']:.5f}\nGRPO {scores['grpo']:.5f}\n"
            f"delta {scores['delta']:+.5f}\n\n"
            + ("Polished weights will be used for decoding."
               if accepted else "Polished weights discarded; the SFT adapter continues."),
        )

    def finished(self, oof: float, baseline: float, path: Path) -> None:
        self.send(
            f"run finished - OOF {oof:.5f}",
            f"out-of-fold score  {oof:.5f}\n"
            f"before fusion      {baseline:.5f}  ({oof - baseline:+.5f} from post-processing)\n"
            f"submission         {path}",
        )

    def failed(self, where: str, detail: str) -> None:
        self.send(f"{where}", f"The run stopped.\n\n{detail}"[:8000], urgent=True)

    def install_crash_handler(self) -> None:
        """Email any uncaught exception, then print it as usual.

        Replaces the bootstrap handler that `run_pipeline.py` installs before this
        object exists, which is why it chains to the interpreter default rather
        than to whatever hook is currently registered - otherwise a crash would be
        reported twice.
        """
        def hook(exc_type, exc, tb):
            self.failed("run crashed",
                        "".join(traceback.format_exception(exc_type, exc, tb)))
            sys.__excepthook__(exc_type, exc, tb)

        sys.excepthook = hook


NOTIFY = Notifier(CFG)
