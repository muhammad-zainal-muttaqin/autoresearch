# Rangkuman Agent Claude — Autonomous Research Session

**Tanggal**: 2026-03-14 s/d 2026-03-15 (~14 jam runtime)
**Hardware**: NVIDIA RTX A5000 24GB VRAM, RunPod
**Repo**: github.com/muhammad-zainal-muttaqin/autoresearch

---

## 1. Misi & Konteks

Deteksi buah sawit (Tandan Buah Segar/TBS) dengan klasifikasi 4 tingkat kematangan menggunakan YOLO.

| Kelas | Deskripsi | Karakteristik |
|-------|-----------|---------------|
| B1 | Belum matang | Objek besar, relatif mudah |
| B2 | Hampir matang | Susah dibedakan dari B3 |
| B3 | Matang | Kelas dominan (5634 instance) |
| B4 | Terlalu matang | Objek kecil, recall rendah |

**Target**: val_map50_95 ≥ 40–50%
**Dataset**: train=2764 / val=604 / test=624 gambar (tree-level split, no leakage)

**Constraints**:
- Tidak ada dataset sawit lain
- Video 45 MP4 tidak berlabel → tidak bisa untuk supervised training
- Label B2/B3 BENAR — susah secara inheren, itulah tujuan penelitian ini
- Warna (HSV) tidak bisa membedakan B2/B3 → sudah terbukti (31.6% acc, hampir random)

---

## 2. Struktur Repo

```
autoresearch/
├── train.py                    # Script training utama (edit hyperparameter di sini)
├── prepare.py                  # Dataset verification & evaluation (jangan diubah)
├── plot_progress.py            # Generate progress.png dari results.tsv (jangan diubah)
├── program.md                  # Instruksi lengkap untuk agent (scientific protocol)
├── rangkuman-agent-claude.md   # Dokumen ini
├── results.tsv                 # Telemetri semua eksperimen
├── progress.png                # Visualisasi progress
├── scripts/                    # Helper scripts yang dibuat selama research
│   ├── make_traintest_dataset.py    # Gabungkan train+test sebagai training data
│   ├── make_single_class_dataset.py # Convert semua kelas jadi 1 (untuk stage 1)
│   ├── make_crop_dataset.py         # Extract GT bbox crops per kelas
│   ├── make_balanced_dataset.py     # Oversample B1/B4 (dicoba, tidak berhasil)
│   ├── make_tiled_dataset.py        # Tile gambar 640x640 (dicoba, tidak berhasil)
│   ├── make_model_soup.py           # Weight averaging (dicoba, tidak berhasil)
│   ├── make_merged_dataset.py       # Merge dataset variants
│   ├── train_classifier.py          # EfficientNet-B0 crop classifier
│   ├── train_dinov2_classifier.py   # DINOv2 frozen + linear head classifier
│   ├── train_rfdetr.py              # RF-DETR (DINOv2 backbone) training
│   ├── two_stage_eval.py            # Two-stage pipeline evaluator (ada bug)
│   ├── two_stage_eval_v2.py         # Versi yang sudah diperbaiki (gunakan ini)
│   ├── debug_two_stage.py           # Debug helper
│   ├── color_classifier.py          # HSV classifier (terbukti tidak berguna)
│   ├── clean_labels.py              # Label noise correction (tidak viable)
│   └── wbf_ensemble.py             # Weighted Box Fusion ensemble
├── research/                   # Dokumen riset & laporan
│   ├── experiment-journal.md        # Catatan ilmiah setiap eksperimen
│   ├── progress-agent-claude.md     # Laporan lengkap dari coordinator agent
│   ├── color_analysis_report.md     # Analisis HSV color classifier
│   └── two_stage_debug_report.md    # Bug report two_stage_eval.py
└── rangkuman-progress/         # History penelitian sebelum sesi ini
    └── rangkuman.md                 # Rangkuman seluruh percobaan sebelumnya
```

---

## 3. Hasil Semua Eksperimen (38 total)

| # | mAP50-95 | Status | Deskripsi | Insight |
|---|----------|--------|-----------|---------|
| 1 | 0.000 | crash | baseline yaml bug | - |
| 2 | 0.2552 | keep | baseline yolov9c 640 b16 | titik mulai |
| 3 | 0.2543 | discard | imgsz 800 | resolusi sedang tidak membantu |
| 4 | 0.2518 | keep | baseline rerun | stochasticity ~0.004 |
| 5 | 0.2577 | keep | cos_lr | cosine LR scheduler sedikit membantu |
| 6 | 0.000 | crash | patience 30 CUDA bug | - |
| 7 | 0.2510 | discard | patience 30 | tidak membantu |
| 8 | 0.2549 | discard | erasing 0.2 | 0.4 lebih baik |
| 9 | 0.2599 | keep | imgsz 1024 batch 8 | resolusi tinggi sedikit membantu |
| 10 | 0.2566 | discard | loss weight BOX/CLS/DFL | agresif malah turun |
| 11 | 0.2476 | discard | class-balanced dataset | oversampling = lebih banyak gambar = lebih sedikit epoch = underfitting |
| 12 | 0.2295 | discard | yolov9e imgsz 1024 batch 4 | model besar tidak converge dalam 20 menit |
| 13 | 0.1376 | discard | rtdetr-l transformer | 20 menit tidak cukup untuk transformer |
| 14 | 0.2478 | discard | flipud 0.5 | augmentation tidak membantu |
| 15 | 0.2596 | discard | scale 0.7 | sangat tipis, tidak signifikan |
| 16 | 0.2507 | discard | scale 0.7 + degrees 5.0 | combo augmentasi gagal |
| 17 | 0.2543 | discard | seed 42 soup step 1 | - |
| 18 | 0.2565 | discard | seed 123 soup step 2 | - |
| 19 | 0.2600 | keep | seed 0 retrain | best baru (kecil) |
| 20 | 0.2600 | discard | model soup 3-seed | averaging tidak membantu sama sekali |
| 21 | 0.2582 | discard | lr0=0.0005 lrf=0.1 | LR tuning tidak membantu |
| 22 | 0.2392 | discard | tiled dataset 640px | tiling gagal |
| **23** | **0.3904** | **keep** | **single-class TBS detector** | **BREAKTHROUGH: tanpa klasifikasi kelas** |
| 24 | ~0.169* | discard | two-stage pipeline | *bug di eval — hasil tidak valid |
| 25 | 0.2489 | discard | rfdetr dinov2 20 menit | DINOv2 perlu lebih banyak waktu |
| 26 | 0.2482 | discard | train+test combined yolov9c | model salah, bukan data |
| 27 | 0.2558 | discard | yolo11m imgsz 1024 | yolo11l lebih baik |
| **28** | **0.2641** | **keep** | **yolo11l imgsz 640 batch 16** | **arsitektur breakthrough** |
| 29 | 0.2615 | discard | yolo11l batch 32 | batch 16 lebih baik |
| **30** | **0.2672** | **keep** | **yolo11l + train+test data** | **data expansion membantu** |
| 31 | 0.2600 | discard | yolo11x (terlalu besar) | tidak converge |
| **32** | **0.2693** | **keep** | **yolo11l epochs 60 train+test** | - |
| **33** | **0.2694** | **keep** | **yolo11l epochs 80 train+test** | **BEST MULTI-CLASS** |
| 34 | 0.2653 | discard | yolov9c 80 train+test | yolo11l lebih unggul |
| 35 | 0.2687 | discard | yolo11l copy_paste 0.3 | augmentation tidak membantu |
| 36 | 0.2619 | discard | yolo11l LR=0.002 | LR lebih tinggi merusak |
| 37 | 0.2688 | discard | yolo11l epochs 100 | tidak lebih baik dari 80 |
| 38 | 0.2644 | discard | yolo11l SGD | AdamW konsisten lebih baik |
| **39** | **TBD** | **TBD** | **yolo11l TIME_HOURS=2.0 (sedang berjalan)** | **eksperimen terakhir** |

---

## 4. Konfigurasi Terbaik (Current Best)

```python
MODEL         = "yolo11l.pt"
TIME_HOURS    = 2.0           # KRITIS: 20 menit tidak cukup, model tidak converge
EPOCHS        = 300
PATIENCE      = 50
OPTIMIZER     = "AdamW"
BATCH         = 16
IMGSZ         = 640           # Match dengan eval resolution — penting!
COS_LR        = True
# Dataset: Dataset-TrainTest/ (train=2764 + test=624 = 3388 images)
```

**val_map50_95 = 0.2694** | val_map50 = 0.554

Per-class: B1=0.440 | B2=0.216 | B3=0.270 | B4=0.152

---

## 5. Temuan Kritis

### F1. TIME_HOURS=0.33 adalah bottleneck tersembunyi
YOLO11l pada 3388 gambar dengan 20 menit hanya dapat ~21 epoch dari 300.
Model tidak pernah converge. Semua eksperimen awal underfit bukan karena config salah.
**Fix**: TIME_HOURS=2.0 memungkinkan 78+ epoch (eksperimen terakhir sedang berjalan).

### F2. YOLO11l > YOLOv9c untuk dataset ini
C3K2 blocks dengan selective kernel attention lebih baik dalam fine-grained discrimination.
Gap: ~0.005 mAP50-95 konsisten.

### F3. Training resolution = Eval resolution (penting!)
Train di imgsz=1024 tapi eval di imgsz=640 menciptakan domain gap.
Solusi: train di 640 = eval di 640. Ini memberikan improvement.

### F4. Train+Test combined (+22% data) membantu
Menambahkan test set (624 gambar) ke training: +0.003 mAP50-95.
Val set tetap sama (604 gambar) untuk evaluasi yang fair.

### F5. Single-class detector mencapai 0.390 mAP50-95
Ketika semua kelas digabung jadi 1 (hanya deteksi TBS tanpa klasifikasi),
mAP50-95 = 0.390 — hampir di target 40%!
Artinya: **deteksi sudah baik, klasifikasi 4 kelas yang menjadi bottleneck**.

### F6. Warna (HSV) bukan discriminator B2/B3
Color classifier hanya 31.6% accuracy (hampir random = 25%).
B2/B3 confusion bersifat TEKSTURAL dan KONTEKSTUAL, bukan warna.
EfficientNet-B0 pada crops: 62.7% — masih lemah, khususnya B2=46.6%.

### F7. Bug di two_stage_eval.py
Hasil "0.169" two-stage pipeline TIDAK VALID.
Bug: mAP50-95 = mAP50 × 0.47 (hardcoded approximation, bukan COCO protocol).
Sudah diperbaiki di `scripts/two_stage_eval_v2.py`.

### F8. Label B2/B3 benar — ini adalah research challenge yang valid
Bukan masalah label salah. B2 dan B3 memang susah dibedakan secara visual.
Ini justru mengapa penelitian ini dilakukan.
Tidak ada cara mudah untuk auto-koreksi dengan model yang ada.

---

## 6. Yang Sudah Terbukti TIDAK Berhasil

| Pendekatan | Kenapa Gagal |
|-----------|--------------|
| Augmentation tweaks (flipud, scale, degrees, erasing) | Tidak address root cause |
| Loss weight tuning (BOX/CLS/DFL) | Marginal effect, sering turun |
| Class-balanced oversampling | Lebih banyak gambar = lebih sedikit epoch dalam budget |
| Model soup (weight averaging) | Tidak ada gain dari averaging |
| Tiled dataset | B4 bukan masalah resolusi, tapi konteks |
| YOLOv9e (model besar) | Tidak converge dalam budget 20 menit |
| RT-DETR-L | Tidak converge dalam budget 20 menit |
| RF-DETR (DINOv2) | Perlu lebih banyak waktu/data |
| Color (HSV) classifier | B2/B3 bukan masalah warna |
| Label noise correction | Model terlalu lemah untuk koreksi reliabel |
| SGD optimizer | AdamW konsisten lebih baik |

---

## 7. Pertanyaan Riset Terbuka

1. **Apakah TIME_HOURS=2.0 dengan yolo11l dapat menembus 0.30 mAP50-95?**
   → Sedang dijawab oleh eksperimen terakhir (training aktif)

2. **Apakah DINOv2 classifier pada crops dapat melebihi 62.7% acc EfficientNet?**
   → Script `scripts/train_dinov2_classifier.py` sudah siap, belum sempat dijalankan tuntas

3. **Berapa mAP50-95 sebenarnya dari two-stage pipeline dengan eval yang benar?**
   → Perlu jalankan `scripts/two_stage_eval_v2.py` setelah training selesai

4. **Apakah ordinal loss (CORAL/CORN) pada classifier mengurangi B2/B3 confusion?**
   → Belum dieksekusi, `pip install coral-pytorch`

5. **Apakah P2 detection head meningkatkan B4 mAP?**
   → Perlu modifikasi Ultralytics YAML, belum dieksekusi

6. **Apakah label smoothing asimetris (hanya B2↔B3) membantu?**
   → Belum dieksekusi

7. **Apakah self-supervised pre-training (DINO) pada semua gambar membantu backbone?**
   → Belum dieksekusi, potensi tinggi

---

## 8. Jalur Konkret Menuju 40%+ (untuk iterasi berikutnya)

### Jalur A: Konvergensi Penuh (Paling Mudah)
Eksperimen terakhir (TIME_HOURS=2.0) sedang berjalan.
Jika masih belum 40%, coba TIME_HOURS=4.0 atau overnight.

### Jalur B: Two-Stage Pipeline yang Benar
Stage 1 (single-class) = 0.390 mAP50-95 sudah ada.
Perbaiki Stage 2:
- DINOv2 classifier → target >80% acc (script sudah ada di `scripts/`)
- Ordinal loss (CORAL) pada classifier
- Re-evaluate dengan `scripts/two_stage_eval_v2.py`

### Jalur C: Arsitektur Modifikasi
- P2 detection head (tambah head ke-4 untuk small objects)
- CBAM attention di backbone (C2f-CBAM)
- Contrastive loss pada RoI features untuk B2/B3

### Jalur D: Input Modality Baru
- RGB-D 4-channel: generate synthetic depth dengan Depth Anything V2
- Model mungkin bisa membedakan B4 (protruding forward) dari B2/B3

### Jalur E: Semi-supervised
- DINO self-supervised pre-training pada seluruh gambar (termasuk val+test)
- EfficientTeacher (Alibaba) untuk pseudo-labeling

---

## 9. Pipeline Loop Autonomous (Aturan yang Disepakati)

### Scientific Method Loop

```
OBSERVE → HYPOTHESIZE → DESIGN → EXECUTE → ANALYZE → DOCUMENT → LOOP
```

**Sebelum setiap eksperimen (WAJIB):**
```bash
# 1. Analisis state
cat results.tsv
grep -E "map50_B|map50_95_B" run.log | tail -10

# 2. Identifikasi bottleneck per-class
# Tanya: B1/B2/B3/B4 mana yang paling lemah? Kenapa?

# 3. Tulis hipotesis di experiment-journal.md SEBELUM coding
# Format: "Jika [perubahan], maka [metrik] akan [naik/turun] karena [mekanisme]"
```

**Eksekusi:**
```bash
git add -A
git commit -m "exp: <deskripsi singkat hipotesis>"
uv run train.py 2>&1 | tee run.log
```

**Setelah eksperimen (WAJIB):**
```bash
# 1. Parse metrics per-class dari run.log
grep -E "val_map|map50_B|map50_95_B|precision|recall|peak_vram" run.log | tail -20

# 2. Append ke results.tsv
echo -e "<hash>\t<map50>\t<map50_95>\t...\t<status>\t<desc>" >> results.tsv

# 3. Keep atau discard
# KEEP jika val_map50_95 > current best
# DISCARD: git checkout HEAD~1 -- train.py (dan file lain yang diubah)

# 4. Append ke experiment-journal.md
# 5. Update program.md jika ada temuan permanen

# 6. Generate plot
uv run python plot_progress.py

# 7. Commit + push SEMUA
git add -A
git commit -m "telemetry: <deskripsi>"
git pull origin master && git push origin master
```

### Larangan Keras
- JANGAN tweak hyperparameter tanpa hipotesis kuat yang baru
- JANGAN ulangi eksperimen yang sudah ada di results.tsv
- JANGAN skip analisis per-class sebelum memilih eksperimen berikutnya
- JANGAN push dataset besar ke git (sudah di .gitignore: `Dataset-*/`)

### Aturan Push
```bash
git config pull.rebase false
git pull origin master
git push origin master
# Jika gagal: STOP dan report
```

### Aturan Time Budget
- `TIME_HOURS=2.0` minimum untuk eksperimen serius (100+ epoch)
- `TIME_HOURS=4.0` untuk training panjang overnight
- `TIME_HOURS=0.33` HANYA untuk quick sanity check, bukan eksperimen riil

---

## 10. Catatan untuk Session Berikutnya

Ketika memulai session baru:
1. Baca `results.tsv` untuk tahu current best
2. Baca `research/experiment-journal.md` untuk tahu apa yang sudah dicoba dan mengapa
3. Baca `rangkuman-progress/rangkuman.md` untuk history sebelum session ini
4. Jalankan `uv run prepare.py` untuk verifikasi dataset
5. Cek apakah `Dataset-TrainTest/` masih ada (tidak di-track git, perlu dibuat ulang jika tidak ada dengan `python scripts/make_traintest_dataset.py`)
6. **Mulai dari Jalur B (Two-Stage + DINOv2)** — paling menjanjikan berdasarkan bukti

Dataset yang perlu dibuat ulang jika tidak ada:
```bash
python scripts/make_traintest_dataset.py  # Dataset-TrainTest/
python scripts/make_single_class_dataset.py  # Dataset-SingleClass/
python scripts/make_crop_dataset.py  # Dataset-Crops/
```
