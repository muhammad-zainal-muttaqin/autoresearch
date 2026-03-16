#!/usr/bin/env bash
# run_e0_with_push.sh — run E0 protocol and push results after each phase
set -uo pipefail

export PATH="$HOME/.local/bin:$PATH"
REPO="/workspace/autoresearch-bbc-v1"
BRANCH="bbc-autoresearch-v1"
LOCKFILE="/tmp/e0_running.lock"

# Prevent parallel instances — only one may run at a time
if [ -e "$LOCKFILE" ]; then
  EXISTING_PID=$(cat "$LOCKFILE" 2>/dev/null || echo "unknown")
  echo "[ERROR] Another E0 instance is running (PID $EXISTING_PID). Exiting."
  exit 1
fi
echo $$ > "$LOCKFILE"
trap 'rm -f "$LOCKFILE"' EXIT INT TERM

# Set remote with token
if [ -n "${GITHUB_TOKEN:-}" ]; then
  git -C "$REPO" remote set-url origin \
    "https://${GITHUB_TOKEN}@github.com/muhammad-zainal-muttaqin/autoresearch.git"
fi

cd "$REPO"

push_results() {
  local label="$1"
  echo ""
  echo "=== [PUSH] $label ==="
  uv run python plot_progress.py    2>&1 || true
  uv run python plot_e0_progress.py 2>&1 || true

  git add \
    progress.png e0_progress.png plot_e0_progress.py autopush.sh run_e0_with_push.sh \
    README.md .gitignore experiments/ context/ e0_results/ logs/ 2>/dev/null || true

  if git diff --cached --quiet; then
    echo "[push] nothing new to commit."
    return
  fi
  TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
  git commit -m "chore: E0 results — ${label} — ${TIMESTAMP}"
  git push origin "$BRANCH"
  echo "[push] pushed: $label"
}

get_completed() {
  python3 -c "
import json, pathlib
s = pathlib.Path('e0_results/state.json')
if s.exists():
    d = json.loads(s.read_text())
    print(','.join(d.get('completed_phases', [])))
else:
    print('')
" 2>/dev/null || echo ""
}

echo "=== Starting E0 Protocol with auto-push ==="
PREV_COMPLETED=$(get_completed)

# Run E0 phase by phase, pushing after each new completion
for PHASE in 0A 0B 0C 1B 2 3; do
  # Skip if already completed
  if echo "$PREV_COMPLETED" | grep -q "$PHASE"; then
    echo "[skip] Phase $PHASE already completed"
    continue
  fi

  echo ""
  echo "=========================================="
  echo "  Running Phase $PHASE"
  echo "=========================================="
  uv run e0_protocol.py --phase "$PHASE" 2>&1
  RC=$?

  COMPLETED_NOW=$(get_completed)
  if echo "$COMPLETED_NOW" | grep -q "$PHASE"; then
    push_results "Phase-${PHASE}-complete"
  else
    echo "[warn] Phase $PHASE did not complete (rc=$RC), still pushing partial results"
    push_results "Phase-${PHASE}-partial"
  fi
done

echo ""
echo "=== E0 Protocol finished. Final push. ==="
push_results "E0-final"
echo "All done."
