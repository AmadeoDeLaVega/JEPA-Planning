#!/usr/bin/env bash
# Overnight PushT linear-head run on one machine (written for an M2 Mac / MPS).
# Renders and encodes the goal groups, then fits the head one rank per process,
# so every finished rank is saved even if a later one fails. Re-running skips
# finished steps; a crashed rank's partial folder is moved aside.
#
# Usage (repository root):  caffeinate -dimsu bash code/scripts/run_pusht_linear_head_local.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
DEVICE="${DEVICE:-mps}"
DATA=results/pusht_goal_groups
OUT=results/pusht_linear_head_mac
CONFIG=code/configs/linear_head_training_mac.yaml
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/progress.log"; }

# Render and encode run as separate processes: decord (model) and pygame (env)
# bundle clashing SDL2 libraries on macOS.
for split in train val test; do
  if [[ -f "$DATA/$split/summary.json" ]]; then
    log "render $split: already done"
  else
    log "render $split: start"
    "$PY" code/experiments/make_pusht_goal_groups.py --split "$split" --output "$DATA/$split" \
      >"$OUT/render_$split.log" 2>&1 || { log "render $split: FAILED (see render_$split.log)"; exit 1; }
    log "render $split: done ($(grep -o '"num_images": [0-9]*' "$DATA/$split/summary.json"))"
  fi
  if [[ -f "$DATA/$split/features.pt" ]]; then
    log "encode $split: already done"
  else
    log "encode $split: start"
    "$PY" code/experiments/encode_goal_groups.py "$DATA/$split" --device "$DEVICE" \
      >"$OUT/encode_$split.log" 2>&1 || { log "encode $split: FAILED (see encode_$split.log)"; exit 1; }
    log "encode $split: done"
  fi
done

ranks=$("$PY" -c "import yaml; print(' '.join(map(str, yaml.safe_load(open('$CONFIG'))['ranks'])))")
for rank in $ranks; do
  dir="$OUT/rank$rank"
  if [[ -f "$dir/metadata.json" ]]; then log "rank $rank: already done"; continue; fi
  [[ -e "$dir" ]] && mv "$dir" "$dir.failed-$(date +%s)"
  log "rank $rank: start"
  "$PY" code/experiments/train_linear_head.py --config "$CONFIG" --ranks "$rank" --device "$DEVICE" \
    --train "$DATA/train" --val "$DATA/val" --output "$dir" >"$OUT/train_rank$rank.log" 2>&1 \
    && log "rank $rank: done ($("$PY" -c "import json; m=json.load(open('$dir/metadata.json')); print(m['chosen'], round(m['elapsed_seconds']/60), 'min')"))" \
    || log "rank $rank: FAILED (see train_rank$rank.log)"
done
log "all done"
