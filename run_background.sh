#!/usr/bin/env bash
# Launch the pipeline detached, so it survives the SSH session ending.
#
#   ./run_background.sh --folds 0 1 2 3 4 --hf-repo me/road-htr
#
# Any arguments are forwarded to run_pipeline.py (see --help). Prints the PID and
# the log path, then returns immediately.
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f run_pipeline.py ]]; then
  echo "run_pipeline.py missing - generating it"
  python build/build_script.py
fi

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="logs/nohup_${STAMP}.out"

echo "checking the notification channel ..."
python build/check_notifications.py || echo "  (continuing without email)"

nohup python -u run_pipeline.py "$@" > "$OUT" 2>&1 &
PID=$!
echo "$PID" > logs/run.pid

cat <<EOF

started pid $PID
  console log   $OUT
  detailed log  logs/run_${STAMP}*.log
  follow        tail -f $OUT
  metrics       tensorboard --logdir artifacts/runs --port 6006
  stop          kill \$(cat logs/run.pid)

You will receive email at each checkpoint upload, each GRPO verdict, on completion,
and on any failure.
EOF
