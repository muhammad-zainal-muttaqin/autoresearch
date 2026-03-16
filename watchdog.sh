#!/usr/bin/env bash
# watchdog.sh — monitor E0 training, restart if stuck or dead, push progress
# Called by cron every 10 minutes.
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
# GITHUB_TOKEN must be set in the environment (e.g. via cron or shell profile)

REPO="/workspace/autoresearch-bbc-v1"
BRANCH="bbc-autoresearch-v1"
LOCKFILE="/tmp/e0_running.lock"
WATCHDOG_LOG="$REPO/logs/watchdog.log"
STUCK_SENTINEL="/tmp/e0_last_activity"
STUCK_THRESHOLD=1800   # 30 menit tanpa aktivitas = stuck
OOM_THRESHOLD=23500    # MiB — jika VRAM > ini, potensi OOM

mkdir -p "$REPO/logs"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$WATCHDOG_LOG"
}

push_if_changed() {
  cd "$REPO"
  git remote set-url origin "https://${GITHUB_TOKEN}@github.com/muhammad-zainal-muttaqin/autoresearch.git" 2>/dev/null || true
  uv run python plot_e0_progress.py 2>/dev/null || true
  git add e0_progress.png e0_results/ logs/ 2>/dev/null || true
  if ! git diff --cached --quiet; then
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
    git commit -m "chore: watchdog push — ${TIMESTAMP}" 2>/dev/null
    git push origin "$BRANCH" 2>/dev/null && log "Pushed to GitHub"
  fi
}

start_e0() {
  log "Starting E0 protocol in background..."
  cd "$REPO"
  nohup bash run_e0_with_push.sh >> "$REPO/logs/e0_run.log" 2>&1 &
  log "Started PID=$!"
  touch "$STUCK_SENTINEL"
}

# ── Check GPU VRAM ──────────────────────────────────────────────────────────
VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "0")
GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || echo "0")
log "GPU: ${VRAM_USED}MiB used, ${GPU_UTIL}% util"

# OOM guard
if [ "${VRAM_USED:-0}" -gt "$OOM_THRESHOLD" ] 2>/dev/null; then
  log "WARNING: VRAM ${VRAM_USED}MiB > threshold ${OOM_THRESHOLD}MiB — monitoring closely"
fi

# ── Check if E0 process is alive ────────────────────────────────────────────
E0_ALIVE=0
if [ -f "$LOCKFILE" ]; then
  E0_PID=$(cat "$LOCKFILE" 2>/dev/null || echo "0")
  if [ -n "$E0_PID" ] && kill -0 "$E0_PID" 2>/dev/null; then
    E0_ALIVE=1
    log "E0 process alive (PID=$E0_PID), GPU=${GPU_UTIL}%"
  else
    log "Lock file exists but PID=$E0_PID is dead — cleaning up"
    rm -f "$LOCKFILE"
  fi
fi

# ── Check if process is stuck (alive but GPU idle too long) ─────────────────
if [ "$E0_ALIVE" -eq 1 ]; then
  NOW=$(date +%s)

  # Check if e0_results/ has been modified recently
  LAST_MODIFIED=$(find "$REPO/e0_results" -newer "$STUCK_SENTINEL" -type f 2>/dev/null | wc -l)
  if [ "$LAST_MODIFIED" -gt 0 ]; then
    touch "$STUCK_SENTINEL"
    log "Activity detected ($LAST_MODIFIED new files) — not stuck"
  else
    SENTINEL_AGE=$(( NOW - $(stat -c %Y "$STUCK_SENTINEL" 2>/dev/null || echo "$NOW") ))
    log "No new files since last check. Sentinel age: ${SENTINEL_AGE}s"

    if [ "$SENTINEL_AGE" -gt "$STUCK_THRESHOLD" ] && [ "${GPU_UTIL:-0}" -lt 5 ] 2>/dev/null; then
      log "STUCK DETECTED: ${SENTINEL_AGE}s idle, GPU=${GPU_UTIL}% — killing and restarting"
      kill "$E0_PID" 2>/dev/null || true
      sleep 5
      rm -f "$LOCKFILE"
      E0_ALIVE=0
      push_if_changed
    fi
  fi
fi

# ── Restart if dead ─────────────────────────────────────────────────────────
if [ "$E0_ALIVE" -eq 0 ]; then
  # Check if E0 is actually complete (all phases done)
  COMPLETED=$(python3 -c "
import json, pathlib
s = pathlib.Path('$REPO/e0_results/state.json')
if s.exists():
    d = json.loads(s.read_text())
    phases = d.get('completed_phases', [])
    print(','.join(phases))
" 2>/dev/null || echo "")

  ALL_PHASES="0A,0B,0C,1B,2,3"
  ALL_DONE=1
  for p in 0A 0B 0C 1B 2 3; do
    if ! echo "$COMPLETED" | grep -q "$p"; then
      ALL_DONE=0
      break
    fi
  done

  if [ "$ALL_DONE" -eq 1 ]; then
    log "All E0 phases complete: $COMPLETED — nothing to restart"
    push_if_changed
  else
    log "E0 not complete (done: $COMPLETED) — restarting"
    push_if_changed
    sleep 3
    start_e0
  fi
fi

log "Watchdog check done."
