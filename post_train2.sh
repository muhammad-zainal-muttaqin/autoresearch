#!/bin/bash
set -e
cd /workspace/autoresearch

echo "Waiting for python3 train.py (PID 69453)..."
while kill -0 69453 2>/dev/null; do
    sleep 30
done
echo "Training process 69453 finished!"
sleep 5  # let log flush

# Parse with tr to strip ANSI/control chars
LOG=$(cat run.log | tr -d '\r\033' | sed 's/\[[0-9;]*[mKHJF]//g')

VAL_MAP50=$(echo "$LOG"    | grep -oP "val_map50:\s+\K[0-9]+\.[0-9]+" | tail -1)
VAL_MAP50_95=$(echo "$LOG" | grep -oP "val_map50_95:\s+\K[0-9]+\.[0-9]+" | tail -1)
PRECISION=$(echo "$LOG"    | grep -oP "precision:\s+\K[0-9]+\.[0-9]+" | tail -1)
RECALL=$(echo "$LOG"       | grep -oP "recall:\s+\K[0-9]+\.[0-9]+" | tail -1)
PEAK_VRAM=$(echo "$LOG"    | grep -oP "peak_vram_mb:\s+\K[0-9]+\.[0-9]+" | tail -1)

echo "Parsed:"
echo "  val_map50=$VAL_MAP50"
echo "  val_map50_95=$VAL_MAP50_95"
echo "  precision=$PRECISION"
echo "  recall=$RECALL"
echo "  peak_vram_mb=$PEAK_VRAM"

if [ -z "$VAL_MAP50_95" ]; then
    echo "ERROR: Could not parse results from log!"
    # Dump last 200 printable lines for debug
    echo "$LOG" | tail -200 > /workspace/autoresearch/parse_debug.txt
    echo "Saved parse_debug.txt"
    exit 1
fi

MEMORY_GB=$(python3 -c "print(f'{float(\"$PEAK_VRAM\")/1024:.1f}')")
COMMIT=$(git rev-parse --short HEAD)

BEST_MAP=$(grep "keep" results.tsv | awk -F'\t' '{print $3}' | sort -n | tail -1)
if python3 -c "exit(0 if float('$VAL_MAP50_95') > float('$BEST_MAP') else 1)"; then
    STATUS="keep"
    LABEL="NEW BEST!"
else
    STATUS="discard"
    LABEL="no improvement"
fi
echo "Status: $STATUS ($VAL_MAP50_95 vs best $BEST_MAP — $LABEL)"

DESC="yolo11l TIME=2h epochs=300 patience=50 AdamW train+test FINAL ($STATUS)"
echo -e "$COMMIT\t$VAL_MAP50\t$VAL_MAP50_95\t$PRECISION\t$RECALL\t$MEMORY_GB\t$STATUS\t$DESC" >> results.tsv
echo "Appended to results.tsv"

# Regenerate plot
/home/researcher/.local/bin/uv run python3 plot_progress.py && echo "Plot updated" || echo "Plot update skipped (uv not found)"

git add results.tsv progress.png 2>/dev/null || git add results.tsv
git commit -m "$(cat <<GITMSG
exp: FINAL — yolo11l 2h TIME train+test — val_map50_95=$VAL_MAP50_95 [$STATUS]

Final autonomous research session experiment.
Model: yolo11l, Dataset: Dataset-TrainTest (3388 imgs)
TIME_HOURS=2.0, EPOCHS=300, PATIENCE=50, AdamW, COS_LR=True
BATCH=16, IMGSZ=640, SEED=0

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
GITMSG
)"

git push origin master && echo "Pushed to GitHub!"
echo "All done."
