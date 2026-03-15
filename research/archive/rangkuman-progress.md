# Rangkuman Lengkap Seluruh Percobaan

Dokumen ini menggabungkan rangkum1.md sampai rangkum5.md tanpa kehilangan konteks. Disusun ulang secara kronologis agar mudah dibaca sebagai satu narasi.

---

## Bagian A: Fondasi Dataset (dari rangkum3.md — Repo Dataset-Sawit)

### A1. Pengumpulan data lapangan

- 45 video MP4 untuk pohon Damimas nomor 0810-0854
- Metadata lapangan (keliling pohon, ukuran buah) disimpan di `Video/Information.md`
- 2 varietas: Damimas (A21B) dan Lonsum (A21A)

### A2. Foto multi-sisi per pohon

- Damimas: 3.596 foto
- Lonsum: 396 foto
- Total pohon: 953
- Mayoritas pohon punya 4 sisi, Damimas 0810-0854 punya 8 sisi

### A3. Anotasi manual tingkat kematangan TBS

- Format: LabelMe (JSON)
- 4 kelas: B1, B2, B3, B4
- Total: 17.992 bounding box (Damimas 16.970, Lonsum 1.022)
- B3 paling dominan, terutama di Lonsum

### A4. Sampel negatif (tanpa objek)

- 80 foto tanpa anotasi (Damimas 45, Lonsum 35)
- Dipakai sebagai background/negative samples untuk training YOLO
- Script: `Script/add_negatives.py`

### A5. Pembersihan data dan perbaikan identitas pohon

- Duplikasi ID pada K2 dipindah ke rentang 103-139
- Duplikasi ID pada K4 dipindah ke rentang 137-228
- Gap nomor K3 dirapikan dengan renumber
- Salah varietas `DAMIMAS_A21A` dikoreksi menjadi `LONSUM_A21A`

### A6. Standarisasi nama file skala penuh

- Format lama: `{VARIETAS}_{BLOK}_{KELOMPOK}_{NOMOR}_{SISI}.ext`
- Format baru: `{VARIETAS}_{BLOK}_{NOMOR-BARU}_{SISI}.ext`
- Video: `VID_20260205_{HHMMSS}.mp4` -> `DAMIMAS_A21B_{NOMOR}.mp4`
- Script: `Script/rename_dataset.py` (mode --dry-run, --execute, --rollback)
- Total mapping: 7.949 (3.992 JPG, 3.912 JSON, 45 MP4)

### A7. Ekspor ke format YOLO

- `Script/convert_to_yolo.py` dan `Script/add_negatives.py`
- Split dilakukan di level pohon untuk mencegah data leakage
- Citra `Unlabeled` ditambahkan sebagai negative samples dengan file label kosong

---

## Bagian B: Percobaan Training Awal di Repo Dataset-YOLO (dari rangkum4.md)

### B1. Dataset yang dipakai

| Split | Gambar | Label | Instance |
|---|---:|---:|---:|
| Train | 2776 | 2776 | 12494 |
| Val | 816 | 816 | 3647 |
| Test | 400 | 400 | 1849 |

Kelas: B1, B2, B3, B4

### B2. Pola umum semua run

- Framework: Ultralytics 8.4.14
- Python 3.12.12, PyTorch 2.8.0+cu126
- Device: 2x Tesla T4 dengan DDP, AMP aktif
- Stage 1: 100 epochs, imgsz=640, batch=16, MuSGD, lr0=0.01, patience=20
- Stage 2: fine-tune dari best.pt stage 1, 20 epochs, lr0=0.001, mosaic=0.0

### B3. Tiga percobaan training yang terekam

| Model | Stage 1 Val mAP50 | Stage 2 Val mAP50 | Test mAP50 | Test mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO26n | 0.516 | 0.513 | 0.489 | 0.236 |
| YOLO26s | 0.514 | 0.528 | 0.502 | 0.240 |
| YOLO26m | 0.511 | 0.518 | 0.498 | 0.243 |

- **YOLO26s** adalah hasil terbaik: val mAP50 0.528, test mAP50 0.502
- YOLO26m tidak cukup unggul untuk membenarkan ukurannya

### B4. Performa per kelas (konsisten di semua model)

| Kelas | Val mAP50 (stage 2) | Test mAP50 | Interpretasi |
|---|---:|---:|---|
| B1 | 0.805 - 0.819 | 0.788 - 0.797 | Paling mudah |
| B2 | 0.389 - 0.421 | 0.316 - 0.325 | Lemah |
| B3 | 0.535 - 0.554 | 0.521 - 0.546 | Sedang |
| B4 | 0.308 - 0.320 | 0.326 - 0.349 | Paling sulit |

### B5. Export ke format deploy

| Model | ONNX | SavedModel | TFLite float16 | CoreML |
|---|---:|---:|---:|---:|
| Nano | 9.5 MB | 23.7 MB | 4.8 MB | 4.8 MB |
| Small | 36.5 MB | 91.4 MB | 18.3 MB | 18.3 MB |
| Medium | 78.0 MB | 195.2 MB | 39.1 MB | 39.1 MB |

### B6. Catatan penting

Meskipun dokumentasi menyebut ProgLoss, STAL, mixup, copy_paste, multi_scale, cls=2.5, dfl=3.0, **tidak ada bukti run yang benar-benar memakai setting itu**. Log menunjukkan cls=0.5, dfl=1.5, mixup=0.0, copy_paste=0.0. Percobaan yang terekam masih baseline two-stage MuSGD.

---

## Bagian C: Percobaan Intensif di Repo YOLOBench (dari rangkum1.md)

### C1. Fondasi data dan split

- Total 3.992 gambar, 953 pohon unik, 4 kelas
- Split final: train=2784, val=604, test=604
- Strategi: tree-level grouping + dominant-class stratification

### C2. E0 — Baseline, learning curve, arsitektur, resolusi

#### Learning curve / kecukupan data

16 run: 2 model (YOLOv9s, YOLO26s) x 2 seed (42, 123) x 4 level data (25%, 50%, 75%, 100%)

- Best: YOLOv9s, seed 123, data 100% -> mAP50 = 0.532
- Diminishing returns jelas: 25%->50% gain besar, 75%->100% hampir plateau
- Jumlah data bukan bottleneck utama
- Kelas paling bermasalah: B4
- YOLOv9s konsisten lebih baik dari YOLO26s

#### Sweep model dan resolusi

| Run | Best epoch | Best mAP50 | Catatan |
|---|---:|---:|---|
| yolo9s_640 | 33 | 0.539 | baseline kuat |
| yolo9s_1024 | 32 | 0.541 | salah satu tertinggi |
| yolo9s_1280 | 24 | 0.533 | tidak lebih baik dari 1024 |
| yolo11s_640 | 35 | 0.535 | mendekati v9s |
| yolo11s_1024 | 35 | 0.535 | relatif datar |
| yolo11s_1280 | 25 | 0.527 | turun |
| yolo8s_640 | 17 | 0.524 | di bawah v9s |
| yolo8s_1024 | 25 | 0.522 | tidak unggul |
| yolo8s_1280 | 27 | 0.526 | tetap di bawah v9s |
| yolo26s_640 | 18 | 0.504 | lemah di 640 |
| yolo26s_1024 | 27 | 0.530 | naik di 1024 |
| yolo26s_1280 | 27 | 0.515 | turun lagi |
| yolo9m_1024 | 31 | 0.536 | kuat |
| yolo9m_1024_b16 | 28 | 0.541 | salah satu tertinggi |

Kesimpulan:
- 1024 adalah resolusi paling masuk akal
- 1280 tidak memberi keuntungan berarti
- Ceiling one-stage 4 kelas: sekitar 0.53 - 0.54 mAP50

#### Tuning epoch, patience, batch, LR

| Run | Best epoch | Best mAP50 | Ringkasan |
|---|---:|---:|---|
| yolo9s_640_100e | 46 | 0.538 | hampir sama baseline |
| yolo9s_640_100e_15p | 30 | 0.539 | hampir sama |
| yolo9s_640_300e | 30 | 0.514 | turun |
| yolo26s_640_300e | 20 | 0.468 | turun jauh |
| yolo26s_640_300e_lr0001 | 300 | 0.510 | membaik tapi bukan breakthrough |
| yolo26s_1024_300e | 30 | 0.541 | kuat tapi tetap di ceiling |
| yolo26s_1024_1000e | 29 | 0.526 | training lebih lama tidak membantu |

Menambah epoch tidak otomatis menaikkan performa.

### C3. E1 dan E2 — Quick win

Upaya yang dicoba:
- YOLO11s Obj365 pretrained
- YOLOv8s OIV7 pretrained
- YOLO11s P2-head + light augmentation
- Advanced augmentation (copy_paste, mixup, degrees)
- B2-B4 focused 6-run matrix

Hasil: tidak ada yang menembus ceiling ~53%. Kesimpulan: data quality > model complexity.

### C4. SAHI inference

- mAP50 turun -6.3% dibanding inference native
- Objek di dataset sudah cukup besar, slicing malah menambah false positive
- Keputusan: SAHI tidak dipakai

### C5. Analisis error B2/B3 dan label cleaning

Model audit: YOLOv9m 1024px, val set 604 gambar

Temuan besar:
- Root cause utama: **inkonsistensi label**
- B2 hanya 31.2% benar
- B2 -> B3 confusion: 208 kali
- B3 -> B2 confusion: 85 kali
- 451 gambar kandidat review
- 46 prediksi salah confidence tinggi = kandidat label error terkuat
- 92% masalah terkonsentrasi di estate DAMIMAS
- B4 buruk terutama karena missed detection pada box kecil

**Ini titik balik utama proyek**: fokus beralih dari "cari model terbaik" ke "audit label".

### C6. Hard example mining dan oversampling

Status: AKTIF/PERSIAPAN
- Tahap ekstraksi dan penyusunan dataset hardmine sudah dikerjakan
- Script training hardmine sudah siap
- Belum ada hasil akhir yang membuktikan hardmine mengalahkan baseline

### C7. Two-stage pipeline: detector + classifier

- Stage 1: detector single-class untuk semua tandan
  - Terbaik: MuSGD, mAP50=0.834, mAP50-95=0.389
- Stage 2: classifier crop 224x224 untuk B1-B4
  - Terbaik: top1_acc=0.647, early stop epoch 129, best epoch 79
- Evaluasi E2E final masih menunggu

### C8. Fase 2 training pasca-label-cleaning

Status: PERSIAPAN
- Oversampling B2 dan B4 dengan horizontal flip
- Menaikkan cls loss weight
- Menaikkan copy_paste
- Dirancang detail tapi output akhir belum ada

### C9. Subset cepat + tree rename + anti-dedup

- Rename filename saja tidak cukup menghindari dedup platform
- Solusi: photometric jitter kecil + JPEG re-encode
- Hasil upload: 345 image total (train=112, val=117, test=116), 334 labeled

### C10. Log tambahan E3

| File | Best epoch | Best mAP50 |
|---|---:|---:|
| E3a_CosineLR_1024_seed42 | 22 | 0.531 |
| yolo9m_tuning | 33 | 0.528 |
| E3c | 15 | 0.495 |

Bukan foundation model penuh, lebih ke tuning tambahan.

### C11. Alternate approach

Cabang pemikiran baru:
- Pisahkan sub-problem detection
- Pisahkan sub-problem classification
- Tambahkan counting / deduplication antar view
- Aggregation output akhir per pohon

Status: cabang riset lanjutan, bukan hasil final.

---

## Bagian D: Benchmark Legacy vs V2 di YOLOBench (dari rangkum2.md dan rangkum5.md)

### D1. Fase Legacy: split lama (1-2 Maret 2026)

#### Batch 1 — 2026-03-01

| Run | Model | Scenario | mAP50 | mAP50-95 |
|---|---|---|---:|---:|
| exp1 | YOLO26l | stratifikasi | 0.442 | 0.150 |
| exp2 | YOLO26l | sawit-yolo | 0.447 | 0.164 |
| exp3 | YOLOv9m | stratifikasi | 0.526 | 0.242 |
| exp4 | YOLOv9m | sawit-yolo | 0.613 | 0.315 |

#### Batch 2 — 2026-03-02

| Run | Model | Scenario | mAP50 | mAP50-95 |
|---|---|---|---:|---:|
| exp5 | YOLO26l | damimas-full | 0.594 | 0.314 |
| exp6 | YOLOv9m | damimas-full | 0.573 | 0.218 |
| exp7 | YOLOv9c | damimas-full | **0.650** | **0.328** |
| exp8 | YOLOv9c | stratifikasi | 0.508 | 0.171 |
| exp9 | YOLOv9c | sawit-yolo | 0.522 | 0.185 |

Angka legacy **ter-inflate** oleh split lama yang masih mengandung leakage.

### D2. Titik balik: desain split baru (3 Maret 2026)

Meeting dosen menghasilkan:
- Semua model harus dibandingkan pada test set yang sama
- Train dipisah: all_data, damimas_only, lonsum_only
- 2 model (YOLO26l, YOLOv9c), 2 seed (42, 123)
- Tree-level split wajib

Dataset v2:
- DAMIMAS: 854 pohon, 3596 gambar (train=2504, val=560, test=532)
- LONSUM: 99 pohon, 396 gambar (train=276, val=60, test=60)
- COMBINED: 3992 gambar (train=2780, val=620, test=592)

### D3. Fase V2: tree-level split + shared test (4 Maret 2026)

12 run pada shared combined test set (592 gambar):

| Run | Skenario | Model | Seed | mAP50 | mAP50-95 | P | R |
|---|---|---|---:|---:|---:|---:|---:|
| exp10 | all_data | YOLO26l | 123 | 0.461 | 0.214 | 0.449 | 0.537 |
| exp11 | all_data | YOLO26l | 42 | 0.457 | 0.203 | 0.448 | 0.537 |
| exp12 | all_data | YOLOv9c | 123 | 0.505 | 0.230 | 0.486 | 0.588 |
| exp13 | all_data | YOLOv9c | 42 | 0.504 | 0.226 | 0.482 | 0.611 |
| exp14 | damimas_only | YOLO26l | 123 | 0.469 | 0.220 | 0.454 | 0.539 |
| exp15 | damimas_only | YOLO26l | 42 | 0.465 | 0.203 | 0.446 | 0.547 |
| exp16 | damimas_only | YOLOv9c | 123 | 0.500 | 0.224 | 0.483 | 0.613 |
| exp17 | damimas_only | YOLOv9c | 42 | **0.505** | **0.230** | 0.502 | 0.590 |
| exp18 | lonsum_only | YOLO26l | 123 | 0.232 | 0.091 | 0.313 | 0.294 |
| exp19 | lonsum_only | YOLO26l | 42 | 0.211 | 0.081 | 0.281 | 0.267 |
| exp20 | lonsum_only | YOLOv9c | 123 | 0.257 | 0.091 | 0.289 | 0.300 |
| exp21 | lonsum_only | YOLOv9c | 42 | 0.307 | 0.119 | 0.366 | 0.370 |

**Benchmark V2 terbaik**: exp12/exp17 = YOLOv9c, mAP50=0.505, mAP50-95=0.230

Cross-evaluation legacy vs V2 pada test yang sama:
| Model | Legacy test | V2 combined test |
|---|---:|---:|
| Legacy yv9c_640 | 0.650 | 0.483 |
| V2 damimas_yv9c_42 | 0.599 | **0.505** |

V2 sebenarnya lebih robust, angka legacy tinggi karena setup evaluasi kurang ketat.

### D4. Presentasi ke dosen (5 Maret 2026)

Keputusan: lanjut ke audit dataset untuk mencari sumber error anotasi.

### D5. Audit outlier (6 Maret 2026)

- Total diaudit: 3992 gambar, 17990 bbox
- 4697 pasangan bbox kandidat pelanggaran ordinal
- 120 gambar padat (min 8 bbox)
- Shortlist review: 273 item diperiksa satu per satu

### D6. Finalisasi dataset_cleaned (8 Maret 2026)

- 273 item outlier direview manual 100%
- 42 file label dikoreksi
- 224 file dikonfirmasi aman
- Auto-cleaning drop 10 bbox
- 83 gambar background dipertahankan sebagai negative samples

Dataset final:
- Total gambar: 3992
- Split: train=2780, val=620, test=592
- Total bbox: 17945
- Distribusi: B1=2169, B2=4079, B3=8266, B4=3431

---

## Bagian E: Eksperimen Lanjutan pada dataset_640 (dari rangkum2.md)

| Percobaan | Konfigurasi inti | mAP50 | mAP50-95 | Status |
|---|---|---:|---:|---|
| Baseline terbaik | YOLOv9c, 300 epoch, b16, 640, AdamW | 0.509 | 0.240 | Ada raw log |
| Recipe konservatif | YOLOv9c, 300 epoch, b16, 640, AdamW | 0.500 | 0.234 | Ada raw log |
| Recipe optimizer auto | YOLOv9c, 300 epoch, b16, 640, auto | 0.475 | 0.226 | Ada raw log |
| Stage 1 specialist | YOLOv9c, 140 epoch, b16, 896, AdamW, tiles | 0.504 | 0.250 | Ada raw log |
| Stage 2 context recovery | fine-tune dari Stage 1 | - | ~0.252 | Hanya di result.md |
| HUB loss sweep | sweep box/cls/dfl | - | ~0.238 | Hanya di result.md |
| SAHI inference | sliced inference | lebih buruk | lebih buruk | Kualitatif |

Kesimpulan: baseline sudah cukup kuat, recipe lain hanya naik-turun tipis.

---

## Bagian F: Eksperimen Autoresearch RunPod (repo ini)

### F1. State saat ini

- Branch: `runpod-final`
- Model: YOLOv9c, imgsz 640, batch 16, AdamW, cos_lr=True, erasing=0.4
- Dataset: train=2764, val=604, test=624

### F2. Hasil iterasi

| Commit | val_map50 | val_map50_95 | Memory GB | Status | Description |
|---|---:|---:|---:|---|---|
| 29a11b9 | 0.000 | 0.000 | 0.0 | crash | baseline dataset yaml path bug |
| 9be36b5 | 0.551 | 0.255 | 11.5 | keep | baseline yolov9c 640 b16 |
| 1da12b8 | 0.536 | 0.254 | 14.9 | discard | imgsz 800 |
| 7004205 | 0.532 | 0.252 | 10.0 | keep | baseline rerun yolov9c 640 b16 |
| ef0aeb4 | 0.546 | 0.258 | 10.0 | **keep** | **cos lr** |
| 03d0f7c | 0.000 | 0.000 | 0.0 | crash | patience 30 cuda unavailable |
| 4f1533b | 0.536 | 0.251 | 10.0 | discard | patience 30 |
| bbfe74e | 0.543 | 0.255 | 10.0 | discard | erasing 0.2 |

**Best saat ini**: val_map50_95 = **0.258** (cos lr, commit ef0aeb4)

---

## Kesimpulan Besar dari Seluruh Perjalanan

1. **Ceiling one-stage 4-class**: berulang kali mentok di sekitar 0.53-0.54 mAP50 (~0.25 mAP50-95)
2. **Menambah data, epoch, arsitektur, resolusi, pretrained, P2-head, augmentasi**: belum memberi lonjakan besar
3. **Bottleneck terkuat**: kualitas label (konflik B2 vs B3) dan small object (B4)
4. **YOLOv9c** adalah model paling stabil dan kuat di hampir semua setting
5. **1024** adalah resolusi paling masuk akal (tapi belum dicoba di autoresearch RunPod)
6. **Benchmark V2 yang fair**: mAP50=0.505, mAP50-95=0.230
7. **Autoresearch RunPod best**: val_map50_95=0.258 (sudah lebih tinggi dari V2 benchmark)
8. Arah riset bergeser dari pencarian model ke **audit data dan diagnosis error**
9. **Two-stage pipeline** dan **alternate approach** (counting/dedup per pohon) muncul sebagai kandidat solusi serius
10. **Label cleaning sudah dilakukan** tapi retraining pada dataset_cleaned belum selesai terdokumentasi

## Yang Belum Terselesaikan

- Hardmine training final vs baseline
- Fase 2 oversampling B2+B4 + cls weight + copy_paste
- Gate 1 end-to-end two-stage evaluation
- Foundation model (DINOv2 distillation, GroundingDINO auto-annotation)
- Training pada dataset_cleaned
- Resolusi 1024 di autoresearch RunPod
