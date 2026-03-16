# Automation System — BBC Oil Palm FFB Detection

Folder ini adalah pusat dokumentasi sistem otomasi E0 baseline protocol.
Script-script berada di root repo (`/workspace/autoresearch-bbc-v1/`) karena menggunakan hardcoded path.

---

## Daftar Script

| Script | Lokasi | Fungsi |
|--------|--------|--------|
| `run_e0_with_push.sh` | root | Runner utama E0 — jalankan semua fase sequential, push setelah tiap fase |
| `watchdog.sh` | root | Monitor proses E0 — restart jika mati/stuck, push progress |
| `autopush.sh` | root | Push manual — regenerate chart lalu commit & push |
| `e0_protocol.py` | root | Protokol E0 baseline (EDA → resolusi → learning curve → arch sweep → hyperparam → final) |
| `plot_e0_progress.py` | root | Generate `e0_progress.png` Karpathy-style (gray dots semua run, green step-line untuk improvement) |
| `backfill_metrics.py` | root | Re-evaluate run yang selesai training tapi metriknya NaN/missing |

---

## Arsitektur Sistem

```
cron (setiap 10 menit)
    └── watchdog.sh
            ├── cek GPU util + lock file
            ├── jika STUCK (GPU idle > 30 menit): kill + restart
            ├── jika DEAD + belum selesai: restart run_e0_with_push.sh
            └── jika DEAD + semua fase done: push final

run_e0_with_push.sh  ← dijalankan oleh watchdog atau manual
    ├── Loop fase: 0A → 0B → 0C → 1B → 2 → 3
    ├── Setiap fase: uv run e0_protocol.py --phase $PHASE
    ├── Setelah tiap fase: regenerate chart + git commit + git push
    └── Lock file: /tmp/e0_running.lock (cegah parallel instance)

e0_protocol.py
    ├── Phase 0A: EDA (distribusi kelas, ukuran gambar)
    ├── Phase 0B: Resolution sweep (640 vs 1024)
    ├── Phase 0C: Learning curve (data fraction)
    ├── Phase 1B: Architecture sweep (yolov8n/s, yolov10n/s, yolo11n/s)
    ├── Phase 2: Hyperparameter optimization (LR, batch, augmentasi)
    └── Phase 3: Final validation (2 seeds, confusion matrix, TFLite export)
```

---

## Cara Pakai

### Jalankan E0 dari awal
```bash
export GITHUB_TOKEN="<token>"
bash run_e0_with_push.sh >> logs/e0_run.log 2>&1 &
```

### Resume E0 jika terhenti
```bash
export GITHUB_TOKEN="<token>"
rm -f /tmp/e0_running.lock        # hapus lock stale jika ada
bash run_e0_with_push.sh >> logs/e0_run.log 2>&1 &
```
Script otomatis skip fase yang sudah ada di `e0_results/state.json`.

### Cek status E0
```bash
# Apakah proses hidup?
kill -0 $(cat /tmp/e0_running.lock) && echo "ALIVE" || echo "DEAD"

# Fase apa yang sudah selesai?
uv run e0_protocol.py --status

# GPU util
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
```

### Push manual (tanpa menjalankan training)
```bash
export GITHUB_TOKEN="<token>"
bash autopush.sh "manual-push"
```

### Backfill metrik run yang NaN
```bash
uv run python backfill_metrics.py
```

### Jalankan fase tertentu saja
```bash
uv run e0_protocol.py --phase 3    # hanya Phase 3
uv run e0_protocol.py --phase 1B   # hanya Phase 1B
```

---

## Setup Cron Watchdog

```bash
crontab -e
```

Tambahkan baris ini (ganti token):
```
GITHUB_TOKEN=<token>
*/10 * * * * bash /workspace/autoresearch-bbc-v1/watchdog.sh >> /workspace/autoresearch-bbc-v1/logs/watchdog.log 2>&1
```

Cek cron aktif:
```bash
crontab -l
```

---

## File State & Output

| Path | Isi |
|------|-----|
| `e0_results/state.json` | State machine: fase selesai, locked imgsz, top archs, best configs |
| `e0_results/results.csv` | Semua metrik tiap run (map50, precision, recall, per-class mAP, dll) |
| `e0_results/runs/` | Artefak training tiap run (weights, plots, args.yaml) |
| `e0_results/reports/` | Laporan Markdown tiap fase |
| `e0_results/plots/` | Plot perbandingan antar fase |
| `e0_progress.png` | Karpathy-style progress chart (auto-update setiap push) |
| `logs/watchdog.log` | Log watchdog |

---

## Keamanan

- **Jangan commit GITHUB_TOKEN** — token harus di-set via environment, bukan hardcode di file.
- Lock file `/tmp/e0_running.lock` mencegah 2 instance berjalan bersamaan (cegah OOM).
- OOM threshold: 23500 MiB (watchdog akan log WARNING jika VRAM mendekati batas).
- GPU model: NVIDIA RTX A5000 (24564 MiB).

---

## Troubleshooting

| Gejala | Solusi |
|--------|--------|
| Lock file stale setelah crash | `rm -f /tmp/e0_running.lock` |
| Phase N tidak muncul di state.json | Edit `state.json` manual, tambahkan phase ke `completed_phases` |
| `top_archs: []` di state.json | Isi manual dari hasil `results.csv` Phase 1B terbaik |
| `locked: {}` di state.json | Set `"locked": {"imgsz": 1024}` berdasarkan hasil Phase 0B |
| Run selesai tapi map50=NaN | Jalankan `uv run python backfill_metrics.py` |
| Push diblokir GitHub Push Protection | Pastikan tidak ada token di file yang di-commit |
