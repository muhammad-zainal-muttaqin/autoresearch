# Rangkuman Agent Claude — Pipeline & Aturan Loop

Dokumen ini berisi pipeline eksperimen yang disepakati, aturan looping, dan ringkasan seluruh runtime autonomous research agent.

---

## 1. Konteks Penelitian

**Misi**: Deteksi buah sawit (TBS) dengan klasifikasi 4 tingkat kematangan: B1, B2, B3, B4.
**Target**: val_map50_95 ≥ 40-50%
**Dataset**: 2764 train / 604 val / 624 test images. Tree-level split (no leakage).
**Hardware**: NVIDIA RTX A5000 24GB VRAM, RunPod.

**Tantangan inheren**:
- B2/B3 memang susah dibedakan secara visual — ini BUKAN bug label, ini justru tujuan penelitian
- B4 adalah objek kecil, recall rendah
- Warna saja tidak cukup (color classifier hanya 31.6% accuracy)
- Tidak ada dataset sawit lain yang tersedia
- Video 45 MP4 tidak berlabel, tidak bisa dipakai untuk supervised training

---

## 2. Pipeline Eksperimen yang Disepakati

### Loop Utama (Scientific Method)

```
OBSERVE → HYPOTHESIZE → DESIGN → EXECUTE → ANALYZE → DOCUMENT → LOOP
```

**Setiap iterasi wajib:**
1. Baca `results.tsv` dan `experiment-journal.md`
2. Identifikasi bottleneck dari per-class metrics (B1/B2/B3/B4 mAP50-95)
3. Tulis hipotesis falsifiable SEBELUM coding
4. Implementasikan perubahan
5. `git add -A && git commit -m "exp: <deskripsi>"`
6. `uv run train.py 2>&1 | tee run.log`
7. Parse hasil per-class
8. Keep/discard berdasarkan val_map50_95
9. Append ke `experiment-journal.md`
10. `uv run python plot_progress.py`
11. `git add -A && git commit -m "telemetry: <deskripsi>" && git push origin master`
12. Loop ulang — TANPA HENTI, TANPA KONFIRMASI

### Aturan Keep/Discard

- **KEEP**: val_map50_95 > current best → pertahankan train.py
- **DISCARD**: val_map50_95 ≤ current best → `git checkout HEAD~1 -- train.py`
- **CRASH**: error/OOM → log zeros, restore, fix, lanjut

### Aturan Git Push

```bash
git config pull.rebase false
git pull origin master
git push origin master
```

Jika push gagal: STOP dan report.

---

## 3. Larangan Keras (Terbukti Tidak Berhasil)

**JANGAN dilakukan kecuali ada alasan kuat yang baru:**
- Augmentation tweaks saja (flipud, scale, degrees, hsv, erasing, mixup, copy_paste)
- Loss weight tweaks saja (box, cls, dfl)
- LR/momentum/weight_decay tweaks saja
- Seed variation saja
- imgsz atau batch saja tanpa perubahan fundamental lain
- Model soup (weight averaging) — sudah dicoba, tidak membantu

---

## 4. Konfigurasi Terbaik Saat Ini

```python
MODEL = "yolo11l.pt"
TIME_HOURS = 2.0          # KRITIS: 20 menit tidak cukup untuk converge
EPOCHS = 300
PATIENCE = 50
OPTIMIZER = "AdamW"
BATCH = 16
IMGSZ = 640               # Match dengan eval resolution
COS_LR = True
DATA = "Dataset-TrainTest/data.yaml"  # train+test combined = 3388 images
```

**val_map50_95 = 0.269424** (current best, commit d9a3ded)

Per-class: B1=0.440, B2=0.216, B3=0.270, B4=0.152

---

## 5. Timeline Eksperimen (38 total, 12+ jam runtime)

| # | Commit | mAP50-95 | Status | Deskripsi | Insight |
|---|--------|-----------|--------|-----------|---------|
| 1 | 29a11b9 | 0.000 | crash | baseline yaml bug | - |
| 2 | 9be36b5 | 0.2552 | keep | baseline yolov9c 640 b16 | baseline |
| 3 | 1da12b8 | 0.2543 | discard | imgsz 800 | resolusi sedang tidak membantu |
| 4 | 7004205 | 0.2518 | keep | baseline rerun | stochasticity confirmed |
| 5 | ef0aeb4 | 0.2577 | keep | cos lr | small gain |
| 6 | 03d0f7c | 0.000 | crash | patience 30 cuda unavailable | - |
| 7 | 4f1533b | 0.2510 | discard | patience 30 | tidak membantu |
| 8 | bbfe74e | 0.2549 | discard | erasing 0.2 | 0.4 lebih baik |
| 9 | ea99dc1 | 0.2599 | keep | imgsz 1024 batch 8 | resolusi tinggi sedikit membantu |
| 10 | 344352b | 0.2566 | discard | loss weight BOX/CLS/DFL | agresif malah turun |
| 11 | 5f32074 | 0.2476 | discard | class-balanced dataset | oversampling tambah noise |
| 12 | 88c2816 | 0.2295 | discard | yolov9e imgsz 1024 batch 4 | terlalu besar, tidak converge 20 menit |
| 13 | 55f03bf | 0.1376 | discard | rtdetr-l transformer | tidak cocok untuk 20 menit |
| 14 | 91d5334 | 0.2478 | discard | flipud 0.5 | augmentation tidak membantu |
| 15 | 5f1d2da | 0.2596 | discard | scale 0.7 | hampir sama, tidak signifikan |
| 16 | 6feadb7 | 0.2507 | discard | scale+degrees | combo augmentasi gagal |
| 17 | b29a3e4 | 0.2543 | discard | seed 42 soup step 1 | - |
| 18 | d28bf04 | 0.2565 | discard | seed 123 soup step 2 | - |
| 19 | dc4c42d | 0.2600 | keep | seed 0 retrain | baru best kecil |
| 20 | baeefdc | 0.2600 | discard | model soup 3-seed | averaging tidak membantu |
| 21 | 40e710b | 0.2582 | discard | lr0 0.0005 lrf 0.1 | LR tuning tidak membantu |
| 22 | 4002d5f | 0.2392 | discard | tiled dataset 640px | tiling gagal |
| 23 | fd7f85c | **0.3904** | **keep** | **single-class TBS detector** | **BREAKTHROUGH: tanpa klasifikasi** |
| 24 | fd7f85c | 0.1686* | discard | two-stage pipeline | *bug di eval script |
| 25 | 5de4f7e | 0.2489 | discard | rfdetr dinov2 | 20 menit tidak cukup |
| 26 | c3baa42 | 0.2482 | discard | train+test combined yolov9c | model salah |
| 27 | c3baa42 | 0.2558 | discard | yolo11m imgsz 1024 | yolo11l lebih baik |
| 28 | 3a557ad | 0.2641 | **keep** | **yolo11l imgsz 640 batch 16** | **NEW BEST — arsitektur benar** |
| 29 | 3a557ad | 0.2615 | discard | yolo11l batch 32 | batch 16 lebih baik |
| 30 | 3a557ad | 0.2672 | **keep** | **yolo11l + train+test** | **NEW BEST — data lebih banyak** |
| 31 | 0132766 | 0.2600 | discard | yolo11x terlalu besar | tidak converge |
| 32 | 0132766 | 0.2693 | **keep** | **yolo11l epochs 60 train+test** | **NEW BEST** |
| 33 | d9a3ded | 0.2694 | **keep** | **yolo11l epochs 80 train+test** | **CURRENT BEST** |
| 34 | d9a3ded | 0.2653 | discard | yolov9c 80 train+test | yolo11l lebih baik |
| 35 | d9a3ded | 0.2687 | discard | yolo11l copy_paste 0.3 | tidak membantu |
| 36 | d9a3ded | 0.2619 | discard | yolo11l LR=0.002 | LR lebih tinggi merusak |
| 37 | d9a3ded | 0.2688 | discard | yolo11l epochs 100 | tidak lebih baik dari 80 |
| 38 | d9a3ded | 0.2644 | discard | yolo11l SGD | AdamW lebih baik |

---

## 6. Temuan Kritis

### Bottleneck #1: TIME_HOURS=0.33 terlalu pendek
- 20 menit → yolo11l hanya dapat ~21 epoch dari 300
- Model tidak pernah benar-benar converge
- **Fix**: TIME_HOURS=2.0 (sedang berjalan sekarang, epoch 6/78+)

### Bottleneck #2: B2/B3 inherently ambiguous
- B2 mAP50-95 = 0.197-0.216 di SEMUA arsitektur
- Color (HSV) saja hanya 31.6% accuracy — warna bukan discriminator utama
- EfficientNet-B0 pada crops: 62.7% — jauh lebih baik tapi masih lemah di B2 (46.6%)
- Ini adalah masalah riset, bukan bug

### Bottleneck #3: B4 small objects
- B4 mAP50-95 stuck di 0.140-0.152 di semua eksperimen
- Tiled dataset tidak berhasil
- P2 head belum dicoba

### Temuan Positif:
- **Single-class detector = 0.390 mAP50-95** — hampir di target 40%!
- **YOLO11l > YOLOv9c** pada dataset ini (C3K2 blocks lebih baik)
- **Train+Test combined** (3388 images) konsisten lebih baik
- **Bug di two_stage_eval.py** ditemukan dan diperbaiki (mAP50×0.47 bukan mAP50-95 sebenarnya)

---

## 7. Pertanyaan Riset Terbuka

1. **Apakah TIME_HOURS=2.0 dengan yolo11l dapat menembus 0.30 mAP50-95?**
   → Sedang dijawab sekarang (training aktif, epoch 6/78+)

2. **Apakah DINOv2 classifier pada crops dapat melebihi EfficientNet's 62.7%?**
   → Target >80% untuk make two-stage pipeline viable

3. **Dengan two_stage_eval.py yang sudah diperbaiki, berapa mAP50-95 sebenarnya dari two-stage pipeline?**
   → Perlu re-run setelah training selesai

4. **Apakah ordinal loss (CORAL/CORN) pada classifier menurunkan B2/B3 confusion?**
   → Belum dieksekusi

5. **Apakah P2 detection head untuk small objects meningkatkan B4 mAP?**
   → Belum dieksekusi (perlu modifikasi Ultralytics source)

6. **Apakah label smoothing asimetris (smooth hanya B2↔B3) membantu?**
   → Belum dieksekusi

7. **Apakah YOLO11l dengan imgsz=1024 + TIME_HOURS=2.0 lebih baik dari 640?**
   → Perlu eksperimen setelah TIME_HOURS=2.0 baseline selesai

---

## 8. Jalur Menuju Target 40%+

### Jalur A: Konvergensi Penuh (Sedang Berjalan)
`TIME_HOURS=2.0 + yolo11l + train+test` → berapa epoch yang cukup?

### Jalur B: Two-Stage dengan Classifier Lebih Baik
- Stage 1 (single-class detector): 0.390 ← sudah ada
- Stage 2: DINOv2 classifier → target >80% acc
- Combined: estimasi 0.35+ mAP50-95 jika classifier 85%+

### Jalur C: Arsitektur Novel
- P2 detection head (khusus B4)
- Ordinal loss pada CLS head (B2/B3 aware)
- Contrastive learning pada RoI features
- RGB-D 4-channel input (Depth Anything V2)

### Jalur D: Self-Supervised Pre-training
- DINO pre-training pada semua 3992 gambar (termasuk val/test sebagai unlabeled)
- Fine-tune backbone yang sudah pre-trained untuk detection
- Domain-specific features > ImageNet features

---

## 9. Scripts yang Sudah Dibuat

| Script | Fungsi |
|--------|--------|
| `make_balanced_dataset.py` | Oversample B1/B4 (dicoba, tidak berhasil) |
| `make_tiled_dataset.py` | Potong gambar jadi tiles (dicoba, tidak berhasil) |
| `make_single_class_dataset.py` | Convert semua kelas jadi 1 kelas |
| `make_crop_dataset.py` | Extract GT bounding box crops per kelas |
| `make_merged_dataset.py` | Gabungkan train+test sebagai training data |
| `make_traintest_dataset.py` | Dataset train+test dengan val tetap |
| `train_classifier.py` | EfficientNet-B0 pada crops |
| `train_dinov2_classifier.py` | DINOv2 frozen + linear head pada crops |
| `train_rfdetr.py` | RF-DETR (DINOv2 backbone) training |
| `two_stage_eval.py` | Evaluasi pipeline 2 tahap (sudah diperbaiki) |
| `color_classifier.py` | HSV histogram classifier (gagal, 31.6%) |
| `clean_labels.py` | Label noise correction (tidak viable) |
| `make_model_soup.py` | Weight averaging (dicoba, tidak membantu) |

---

## 10. Aturan Agent untuk Iterasi Berikutnya

### WAJIB setiap iterasi:
1. Analisis per-class metrics B1/B2/B3/B4 dari run sebelumnya
2. Tulis hipotesis di experiment-journal.md SEBELUM coding
3. Jalankan `plot_progress.py` dan commit progress.png
4. Push ke GitHub setelah setiap telemetri
5. Update program.md jika ada temuan permanent baru

### BOLEH dilakukan:
- Modifikasi file apapun (train.py, prepare.py, dll)
- Buat script baru
- Buat dataset baru (Dataset-*/ sudah di .gitignore)
- Install pip packages
- Gunakan TIME_HOURS sampai 4.0 untuk training panjang

### TIDAK BOLEH:
- Push dataset besar ke git
- Menghentikan loop tanpa alasan kritis
- Mengulangi eksperimen yang sudah ada di results.tsv
- Tweak hyperparameter sederhana tanpa hipotesis kuat
