#!/bin/bash
# Wait for training process to finish, then process results

TRAIN_PID=$(pgrep -f "python3 /workspace/autoresearch/train.py" | head -1)
echo "Waiting for training PID $TRAIN_PID to finish..."

if [ -n "$TRAIN_PID" ]; then
    wait "$TRAIN_PID" 2>/dev/null || true
fi

echo "Training finished. Processing results..."

cd /workspace/autoresearch

# Parse final results from run.log
VAL_MAP50=$(grep "val_map50:" run.log | tail -1 | awk '{print $2}')
VAL_MAP50_95=$(grep "val_map50_95:" run.log | tail -1 | awk '{print $2}')
PRECISION=$(grep "precision:" run.log | tail -1 | awk '{print $2}')
RECALL=$(grep "recall:" run.log | tail -1 | awk '{print $2}')
PEAK_VRAM=$(grep "peak_vram_mb:" run.log | tail -1 | awk '{print $2}')

echo "Results:"
echo "  val_map50:     $VAL_MAP50"
echo "  val_map50_95:  $VAL_MAP50_95"
echo "  precision:     $PRECISION"
echo "  recall:        $RECALL"
echo "  peak_vram_mb:  $PEAK_VRAM"

MEMORY_GB=$(echo "$PEAK_VRAM" | awk '{printf "%.1f", $1/1024}')

COMMIT=$(git rev-parse --short HEAD)
DESC="yolo11l TIME=2h epochs=300 patience=50 AdamW train+test (FINAL)"

# Determine status vs best
BEST_MAP=$(grep "keep" results.tsv | awk -F'\t' '{print $3}' | sort -n | tail -1)
STATUS="discard"
if python3 -c "exit(0 if float('$VAL_MAP50_95') > float('$BEST_MAP') else 1)" 2>/dev/null; then
    STATUS="keep"
    echo "NEW BEST! $VAL_MAP50_95 > $BEST_MAP"
else
    echo "No improvement: $VAL_MAP50_95 <= $BEST_MAP"
fi

# Append to results.tsv
echo -e "$COMMIT\t$VAL_MAP50\t$VAL_MAP50_95\t$PRECISION\t$RECALL\t$MEMORY_GB\t$STATUS\t$DESC" >> results.tsv
echo "Appended to results.tsv"

# Regenerate plot
uv run python3 plot_progress.py && echo "Plot regenerated"

# Commit and push
git add results.tsv progress.png
git commit -m "$(cat <<GITMSG
exp: FINAL — yolo11l 2h TIME_HOURS train+test — val_map50_95=$VAL_MAP50_95 [$STATUS]

Final autonomous research session experiment.
Model: yolo11l, Dataset: Dataset-TrainTest (3388 imgs)
TIME_HOURS=2.0, EPOCHS=300, PATIENCE=50, AdamW, COS_LR=True
BATCH=16, IMGSZ=640, SEED=0

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
GITMSG
)"

git push origin master && echo "Pushed to GitHub"
echo "Done."
