#!/usr/bin/env bash
# autopush.sh — regenerate charts, commit new results, push to origin
# Usage: bash autopush.sh [commit message suffix]
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
REPO="/workspace/autoresearch-bbc-v1"
BRANCH="bbc-autoresearch-v1"

cd "$REPO"

# Regenerate progress charts
uv run python plot_progress.py    2>&1 || true
uv run python plot_e0_progress.py 2>&1 || true

# Stage everything that's tracked (or newly unignored)
git add \
  progress.png \
  e0_progress.png \
  plot_e0_progress.py \
  README.md \
  .gitignore \
  experiments/ \
  context/ \
  e0_results/ \
  logs/ \
  2>/dev/null || true

# Only commit if there's something staged
if git diff --cached --quiet; then
  echo "[autopush] Nothing new to commit."
  exit 0
fi

SUFFIX="${1:-auto}"
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
git commit -m "chore: push results — ${TIMESTAMP} [${SUFFIX}]"
git push origin "$BRANCH"
echo "[autopush] Pushed to $BRANCH at $TIMESTAMP"
