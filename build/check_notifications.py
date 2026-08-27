"""Preflight check for the notification channel.

Run this before launching a long unattended job. It loads `.env` exactly the way
the pipeline does and sends one test message, so a bad app password surfaces in
five seconds rather than four hours into a run.

    python build/check_notifications.py
"""

from __future__ import annotations

import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQUIRED = ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "NOTIFY_EMAIL_TO")


def load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        raise SystemExit(f"no {env} - copy .env.example and fill it in")
    for line in env.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def main() -> int:
    load_env()
    missing = [k for k in REQUIRED if not os.environ.get(k)]
    if missing:
        print(f"FAILED: missing {', '.join(missing)} in .env")
        return 1

    to = os.environ["NOTIFY_EMAIL_TO"]
    msg = EmailMessage()
    msg["Subject"] = "[ROAD-HTR] notification channel test"
    msg["From"] = os.environ["SMTP_USER"]
    msg["To"] = to
    msg.set_content(
        "Notification channel test for the R.O.A.D. Barbados HTR pipeline.\n\n"
        "If you are reading this, an unattended run will email you on:\n"
        "  - run started\n"
        "  - each checkpoint uploaded to the hub, with its validation score\n"
        "  - each GRPO accept/reject verdict, with the delta\n"
        "  - run finished, with the out-of-fold score\n"
        "  - any uncaught failure, with the traceback\n"
    )
    try:
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ["SMTP_PORT"]),
                          timeout=30) as server:
            server.starttls()
            server.login(os.environ["SMTP_USER"], os.environ["SMTP_PASSWORD"])
            server.send_message(msg)
    except smtplib.SMTPAuthenticationError:
        print("FAILED: authentication rejected. Gmail needs a 16-character App "
              "Password from https://myaccount.google.com/apppasswords, not the "
              "account password.")
        return 1
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        return 1

    print(f"OK -> test message sent to {to}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
