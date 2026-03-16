# Autoresearch Agent

You are the research agent for BBC oil palm fruit bunch detection (4-class: B1/B2/B3/B4).

## Before doing anything

1. Read `program.md` — your operating rules and coding patterns
2. Read `context/research_overview.md` — exploration priorities and closed branches
3. Read `context/domain_knowledge.md` — domain knowledge
4. Read `context/e0_results.md` — what E0 techniques are unlocked
5. Read `experiments/state.json` — current baselines and experiment counter
6. Read `experiments/summary.md` — your research notebook
7. Read `experiments/log.md` — recent experiment history
8. Read `experiments/results.tsv` — full metrics ledger

## Your job

Run hypothesis-driven experiments to maximize mAP@0.5. Follow the cycle: **observe → hypothesize → intervene → test**.

- Edit `train.py` for hyperparameter experiments (main track)
- Edit `modeling.py` for model changes (exploration track)
- Edit `pipeline.py` for pipeline changes (exploration track)
- Run experiments with `uv run train.py`
- Never edit `prepare.py` or `orchestrator.py`
- Decision metric: mAP@0.5
- Seeds: 42 for regular, 123 for confirmation reruns

## Typical workflow per experiment

1. State your hypothesis and success criterion
2. Make the minimal code change
3. Run `uv run train.py`
4. Analyze results (all metrics, not just headline mAP)
5. Record findings in `experiments/summary.md`
6. If exploration branch beats main, rerun with seed=123 to confirm

## E0 Protocol (baseline experiments)

E0 is a separate automated script. Run it with:
```bash
uv run e0_protocol.py              # full run
uv run e0_protocol.py --resume     # resume
uv run e0_protocol.py --status     # check progress
```
After E0 completes, update `context/e0_results.md` with the results so E0 techniques become available for agent exploration.

## Commands

```bash
uv run train.py                                              # run one experiment
uv run orchestrator.py decide <exp_id> <KEEP|DISCARD|PARK> <reason>  # override decision
uv run orchestrator.py next-batch                            # advance batch
uv run python plot_progress.py                               # regenerate chart
```

## Constraints

- Small YOLO models only: yolov8n, yolov8s, yolov10n, yolov10s, yolo11n, yolo11s
- Max 40 epochs, patience 15
- VRAM cap: 24GB
- Do not use techniques in E0 scope until results appear in `e0_results.md`
