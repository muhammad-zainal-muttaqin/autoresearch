# Autoresearch Program

Mission:
- maximize `val_map50_95`
- keep the validation split fixed
- keep the agent's live surface narrow

Normal editable files:
- `train.py`
- `research.py`

Frozen operational files:
- `prepare.py`
- `orchestrator.py`

The contract is simple:
- edit `train.py` for metadata and config
- edit `research.py` only for nontrivial structural changes
- run `uv run train.py`
- let `orchestrator.py` handle gates, state, logs, reports, and `results.tsv`

## Source Of Truth

Trust sources in this order:

1. `results.tsv`
2. `logs/`
3. `experiments/summary.md` and `experiments/log.md`
4. `train.py`, `research.py`, `prepare.py`, `orchestrator.py`
5. `archive/`

If prose disagrees with telemetry, telemetry wins.

## Current Facts

- decision metric: `val_map50_95`
- canonical split: train `2764`, val `604`, test `624`
- standard comparable dataset path is frozen in `prepare.py`
- best validated 4-class legacy result: `0.269424`
- main ceiling remains `B2/B3` discrimination

Closed branches:
- long brute-force one-stage training
- crop-only two-stage pipelines
- tiled training as previously implemented
- label smoothing as a standalone fix
- small knob-turning on already closed branches

## Hard Rules

1. Every run must have a title, hypothesis, and success criterion in `train.py`.
2. Normal experiments should only edit `train.py`.
3. Structural experiments may edit `research.py`; that automatically becomes an exploration track unless `TRACK_HINT` says otherwise.
4. `prepare.py` and `orchestrator.py` are frozen during normal operation.
5. Default search budget remains short: `TIME_HOURS <= 0.5`, `EPOCHS <= 40`.
6. Do not change the validation split.
7. Do not report approximated `mAP50-95`.
8. Infrastructure failures are logged, but they do not change scientific conclusions.
9. Keep raw logs in `logs/`; do not delete them after summarizing.
10. If this file conflicts with `results.tsv`, update this file.

## Track Semantics

- `train.py` changed, `research.py` clean -> `main`
- `research.py` changed -> `exploration`
- `train.py` + `research.py` changed -> `exploration`
- `TRACK_HINT` may override auto-classification only when necessary

There may be at most one active exploration branch.

## Runtime Loop

```text
SYNC -> OBSERVE -> HYPOTHESIZE -> EDIT -> RUN -> ANALYZE -> RECORD -> SYNC
```

What the runtime does automatically:
- syntax/import gate
- dataset verification
- smoke gate with a dummy forward pass
- track classification
- append `results.tsv`
- append `experiments/log.md`
- regenerate `experiments/summary.md`
- regenerate current batch report

What the agent still does:
- decide what to try
- write a falsifiable hypothesis
- interpret the result
- choose the next step

## Commands

Run one experiment:

```powershell
uv run train.py
```

Advance to the next batch:

```powershell
uv run orchestrator.py next-batch
```

Regenerate the chart:

```powershell
uv run python plot_progress.py
```

## Defaults

- decision status in v1 is machine-provisional only: `keep`, `discard`, `infra_fail`, `crash`
- `PARK` and manual override are deferred to v2
- group experiments are deferred to v2
- `context.md` is the only human steering file
