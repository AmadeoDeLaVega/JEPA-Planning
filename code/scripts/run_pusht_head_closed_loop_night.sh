#!/usr/bin/env bash
# Overnight PushT goal-head closed loop on the Mac (plan step 6.4), resumable.
# Heads run in the priority order of code/configs/pusht_head_closed_loop_order.txt;
# finished heads (results/pusht_head_closed_loop/test/episodes/<head>.csv) are skipped.
# Start:  caffeinate -dimsu bash code/scripts/run_pusht_head_closed_loop_night.sh
# Stop:   pkill -f run_pusht_head_closed_loop_night.sh; pkill -f run_pusht_head_closed_loop.py
#         (the head in progress is lost; it reruns first next time)
set -u
cd "$(dirname "$0")/../.."
OUT=results/pusht_head_closed_loop/test
LOG=results/pusht_head_closed_loop/progress.log
mkdir -p results/pusht_head_closed_loop
RUNS="results/pusht_linear_head_mac/rank4 results/pusht_linear_head_mac/rank16 results/pusht_linear_head_mac/rank32 results/pusht_linear_head_mac/rank128"
HEADS=$(cat code/configs/pusht_head_closed_loop_order.txt)
for attempt in 1 2 3 4 5; do
  echo "[$(date '+%F %T')] closed loop attempt $attempt" >> "$LOG"
  RES=""; [ -d "$OUT" ] && RES="--resume"
  .venv/bin/python -u code/experiments/run_pusht_head_closed_loop.py --bank results/pusht_head_bank/test --runs $RUNS \
    --heads $HEADS --output "$OUT" --device mps $RES 2>&1 | grep --line-buffered -v "^objc\[" \
    | grep --line-buffered -E "episodes in|skip|Error|Traceback|error" >> "$LOG"
  status=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] exit $status" >> "$LOG"
  [ "$status" = 0 ] && break
  sleep 30
done
echo "[$(date '+%F %T')] closed loop finished" >> "$LOG"
