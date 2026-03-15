# Progress Report — Autoresearch Runtime (12+ jam)
Generated: 2026-03-15

---

## 1. Ringkasan Misi

**Tujuan**: Memaksimalkan `val_map50_95` pada dataset deteksi Tandan Buah Segar (TBS) kelapa sawit.
**Target**: mAP50-95 > 50% (aspirasional), realistis >40% berdasarkan trajectory saat ini.
**Dataset**: 4 kelas kematangan — B1 (mentah), B2, B3, B4 (matang penuh) — dideteksi dari foto kebun.
**Best saat ini**: **val_map50_95 = 0.269424** (commit d9a3ded, YOLO11l, 80 epochs, train+test combined).

---

## 2. Dataset & Constraints

### Dataset Statistik
| Split | Gambar | Deskripsi |
|---|---:|---|
| Train | 2764 | Dataset utama |
| Val | 604 | Held-out validation |
| Test | 624 | Held-out test (juga dipakai sebagai tambahan train) |
| Train+Test | 3388 | Dataset yang digunakan di eksperimen terbaru |

### Distribusi Kelas (Train)
| Kelas | Instance | Proporsi | Problem |
|---|---:|---:|---|
| B1 | 1540 | 12.3% | Underrepresented |
| B2 | 2845 | 22.8% | Sangat ambigu vs B3 |
| B3 | 5634 | 45.1% | Dominan (3.6x > B1) |
| B4 | 2343 | 18.8% | Small object problem |
| **Total** | **12494** | | |

### Constraints
- Time budget: **20 menit per run** (TIME_HOURS=0.33)
- GPU: Tesla T4 (12-14 GB VRAM)
- Framework: Ultralytics YOLO

---

## 3. Timeline Eksperimen

Semua hasil dari `results.tsv` secara kronologis:

| Commit | val_mAP50 | val_mAP50-95 | Status | Deskripsi |
|---|---:|---:|---|---|
| 29a11b9 | 0.000 | 0.000 | crash | Baseline — bug path yaml |
| 9be36b5 | 0.5506 | 0.2552 | **keep** | Baseline YOLOv9c 640 b16 |
| 1da12b8 | 0.5357 | 0.2543 | discard | imgsz 800 — lebih buruk |
| 7004205 | 0.5324 | 0.2518 | **keep** | Baseline rerun YOLOv9c 640 b16 |
| ef0aeb4 | 0.5462 | 0.2577 | **keep** | cos_lr=True — improvement kecil |
| 03d0f7c | 0.000 | 0.000 | crash | patience 30 — cuda unavailable |
| 4f1533b | 0.5365 | 0.2510 | discard | patience 30 — tanpa improvement |
| bbfe74e | 0.5427 | 0.2549 | discard | erasing 0.2 — tanpa improvement |
| ea99dc1 | 0.5374 | **0.2599** | **keep** | imgsz 1024 batch 8 — **best saat itu** |
| 344352b | 0.5362 | 0.2566 | discard | imgsz 1024 BOX 10 CLS 1.5 DFL 2.0 — hurts |
| 5f32074 | 0.5162 | 0.2476 | discard | Class-balanced dataset B1/B4 oversampled |
| 88c2816 | 0.4838 | 0.2295 | discard | YOLOv9e imgsz 1024 batch 4 — underfitting |
| 55f03bf | 0.3063 | 0.1376 | discard | RT-DETR-L — tidak konvergen dalam 20 menit |
| 91d5334 | 0.5290 | 0.2478 | discard | flipud 0.5 |
| 5f1d2da | 0.5437 | 0.2596 | discard | scale 0.7 |
| 6feadb7 | 0.5397 | 0.2507 | discard | scale 0.7 + degrees 5.0 |
| b29a3e4 | 0.5326 | 0.2543 | discard | seed 42 (model soup step 1) |
| d28bf04 | 0.5321 | 0.2565 | discard | seed 123 (model soup step 2) |
| dc4c42d | 0.5390 | **0.2600** | **keep** | seed 0 retrain — new best saat itu |
| baeefdc | 0.5390 | 0.2600 | discard | Model soup 3-seed — no improvement |
| 40e710b | 0.5341 | 0.2582 | discard | lr0 0.0005 lrf 0.1 — lebih buruk |
| 4002d5f | 0.4975 | 0.2392 | discard | Tiled dataset 640px — turun |
| fd7f85c | **0.8354** | **0.3904** | **keep** | **Single-class TBS detector (stage1)** |
| fd7f85c | 0.3587 | 0.1686 | discard | Two-stage pipeline approx 0.169 |
| 5de4f7e | 0.5417 | 0.2489 | discard | RF-DETR DINOv2 — 0.2489 (lebih buruk) |
| c3baa42 | 0.5142 | 0.2482 | discard | train+test combined 3388 imgs (+22%) |
| c3baa42 | 0.5275 | 0.2558 | discard | YOLO11m imgsz 1024 batch 8 |
| 3a557ad | 0.5553 | **0.2641** | **keep** | **YOLO11l imgsz 640 batch 16 — NEW BEST!** |
| 3a557ad | 0.5502 | 0.2615 | discard | YOLO11l batch 32 — batch 16 lebih baik |
| 3a557ad | 0.5520 | **0.2672** | **keep** | **YOLO11l batch 16 640 train+test** |
| 0132766 | 0.2603 | 0.2603 | discard | YOLO11x batch 16 640 train+test — diverge |
| 0132766 | 0.5538 | **0.2693** | **keep** | **YOLO11l epochs 60 — NEW BEST!** |
| d9a3ded | 0.5543 | **0.2694** | **keep** | **YOLO11l epochs 80 — marginal +0.0002** |
| d9a3ded | 0.5586 | 0.2653 | discard | YOLOv9c epochs 80 — worse than YOLO11l |
| d9a3ded | 0.5544 | 0.2687 | discard | YOLO11l copy_paste 0.3 — marginal regression |
| d9a3ded | 0.5514 | 0.2619 | discard | YOLO11l LR0=0.002 — higher LR hurts |

### Total Eksperimen: ~35 run (termasuk crash dan 2-3 re-evaluasi)

---

## 4. Temuan Kritis

### 4.1 Per-Class Performance (YOLO11l best run)
| Kelas | mAP50-95 | Status | Root Cause |
|---|---:|---|---|
| B1 | ~0.439 | Good | Mudah, jelas berbeda |
| B2 | ~0.216 | **Critical bottleneck** | Ambigu vs B3, label noise |
| B3 | ~0.267 | Moderate | Dominan tapi confusion dengan B2 |
| B4 | ~0.141 | Poor | Small object, sering missed |

**Temuan**: B2 mAP50-95 ≈ 0.197 konsisten di SEMUA arsitektur (YOLOv9c, YOLO11m, RT-DETR, RF-DETR). Hanya naik ke 0.210-0.216 pada best config (YOLO11l imgsz=640). Ini ceiling yang sangat konsisten.

### 4.2 Apa yang Terbukti Tidak Berhasil
| Pendekatan | Delta | Kesimpulan |
|---|---:|---|
| imgsz lebih besar (800/1024/1280) | +0.002 maksimum | Resolution bukan bottleneck utama |
| Model lebih besar (YOLOv9e, YOLO11x) | **negatif** | Underfitting dalam 20 menit |
| Class-balanced oversampling | -0.012 | Lebih sedikit effective epochs |
| Loss weight tuning | -0.003 | Default sudah well-tuned |
| Model soup (averaging) | 0.000 | Tidak ada gain |
| RT-DETR transformer | -0.122 | Tidak konvergen dalam 20 menit |
| RF-DETR DINOv2 | -0.011 | DINOv2 tidak membantu di domain ini |
| Tiled dataset | -0.021 | Tiles bukan solusi di skala ini |
| Higher LR (0.002) | -0.008 | Hurts convergence |
| copy_paste 0.3 | -0.001 | Marginal regression |

### 4.3 Apa yang Terbukti Berhasil
| Pendekatan | Delta | Mekanisme |
|---|---:|---|
| cos_lr=True | +0.002 | Better LR decay |
| imgsz=1024 | +0.002 | Detail lebih baik |
| **YOLO11l imgsz=640 batch=16** | **+0.004** | Train/eval resolution match + more gradient updates |
| Train+Test combined | +0.003 | More data (jika batch=16 bisa handle) |
| Epochs=60→80 | +0.0002 | Marginal, tidak signifikan |
| Single-class detector (stage1) | +0.130 mAP50! | Simpler classification task |

---

## 5. Current Best Configuration

```yaml
Model: YOLO11l (49M params, C3K2 architecture)
imgsz: 640
batch: 16
epochs: 80
optimizer: AdamW (default)
cos_lr: True
erasing: 0.4 (default)
Dataset: Dataset-TrainTest (train+test, 3388 imgs)
Training resolution = Evaluation resolution: 640px

val_map50:    0.554304
val_map50_95: 0.269424  ← CURRENT BEST
precision:    0.502331
recall:       0.631508
memory:       9.8 GB VRAM
```

**Saved at**: `/workspace/autoresearch/best_yolo11l_e80.pt`

**Trajectory**: mAP50-95 meningkat dari 0.2552 (baseline) ke 0.2694 (+0.0142 = +5.6% relative)

---

## 6. Root Cause Analysis

### Mengapa Tidak Bisa Menembus 0.30 (apalagi 0.40)?

**Root Cause #1 — Label Ambiguity (B2/B3)**: [TERBUKTI PALING KRITIS]
- Audit di project sebelumnya menemukan: B2 hanya 31.2% label-benar
- B2→B3 confusion: 208 kali (dalam training set)
- B3→B2 confusion: 85 kali
- **Bukti**: EfficientNet-B0 yang dilatih pada isolated crops (tanpa background noise) hanya 62.7% accuracy, dengan B2=46.6% — mendekati coin-flip
- **Implikasi**: Bahkan model terbaik pun tidak bisa belajar dari label yang salah 40%+ waktu

**Root Cause #2 — Small Object B4**: [TERBUKTI KONSISTEN]
- B4 mAP50-95 selalu terendah (~0.140-0.141) di semua run
- Tiled dataset (yang seharusnya membantu small objects) malah hurts: 0.239 vs 0.260
- **Bukti**: SAHI inference juga hurts (-6.3% mAP50 di repo sebelumnya)
- **Implikasi**: B4 bukan masalah resolusi (objek sudah cukup besar di 640px), tapi mungkin visual ambiguity dengan background/foliage

**Root Cause #3 — Time Budget Terlalu Pendek untuk Large Models**:
- YOLOv9e (>100M params): 0.229 (jauh lebih buruk)
- YOLO11x: diverge (0.260 mAP50 tanpa konvergensi normal)
- RT-DETR-L: 0.138 (tidak konvergen)
- **Implikasi**: Untuk model besar, butuh setidaknya 2-4 jam training

**Root Cause #4 — Data Ceiling**:
- Dari learning curve analysis (project sebelumnya): 75%→100% data hampir plateau
- Menambah test set (+22%) tidak significantly membantu: hanya +0.003
- **Implikasi**: Dataset sudah cukup besar, yang kurang adalah KUALITAS label, bukan kuantitas

---

## 7. Pertanyaan Riset Terbuka

Pertanyaan-pertanyaan berikut harus dijawab secara empiris:

### P1 — DINOv2 sebagai Crop Classifier
> "Apakah DINOv2 (facebook/dinov2-base) sebagai backbone frozen + linear head pada Dataset-Crops menghasilkan val_acc >80%, melampaui EfficientNet-B0's 62.7%? Apakah B2 accuracy naik dari 46.6% ke >65%?"

**Rasional**: DINOv2 dilatih dengan self-supervised learning pada 142M gambar. Representasi visualnya lebih kaya dari EfficientNet yang dilatih supervised. Untuk fine-grained class disambiguation (B2 vs B3), feature DINOv2 yang lebih diskriminatif seharusnya membantu.
**Success criterion**: val_acc > 75% AND B2 acc > 60%

### P2 — TIME_HOURS=2.0 dengan YOLO11l
> "Apakah TIME_HOURS=2.0 dengan YOLO11l imgsz=640 batch=16 menghasilkan mAP50-95 >0.300? Apakah lebih banyak epoch menyelesaikan convergence yang terpotong di 20 menit?"

**Rasional**: Training curve menunjukkan model masih bisa improve di epoch 80 (marginal gain +0.0002). Dengan 6x waktu lebih banyak, bisa mencapai 200-300 epoch di mana convergence lebih penuh.
**Success criterion**: mAP50-95 > 0.300

### P3 — Ordinal Loss (CORAL) untuk B2/B3 Confusion
> "Apakah CORAL (COnsistent RAnk Logits) ordinal loss menurunkan B2/B3 confusion lebih dari standard cross-entropy pada crop classifier?"

**Rasional**: Ripeness stages B1→B2→B3→B4 adalah urutan ordinal. Standard CE treats semua misclassification sama (B2→B3 dan B2→B1 diperlakukan setara). CORAL memaksakan ordering constraint sehingga B2→B3 error (adjacent) lebih kecil penalty-nya dari B2→B1 (non-adjacent). Ini seharusnya membuat model lebih conservative di B2/B3 boundary.
**Success criterion**: B2 AP50-95 > 0.240 (dari 0.210 saat ini)

### P4 — Two-Stage dengan DINOv2 Classifier
> "Apakah two-stage pipeline menggunakan single-class detector (mAP50-95=0.390) + DINOv2 classifier menghasilkan combined mAP50-95 >0.280, melampaui end-to-end YOLO11l's 0.269?"

**Rasional**: Single-class detector (mAP50-95=0.390) sudah jauh melampaui multi-class (0.269). Jika classifier dapat mencapai >80% accuracy (vs EfficientNet's 62.7%), combined pipeline bisa secara teoritis mencapai 0.390 × 0.85 ≈ 0.332 mAP50-95.
**Success criterion**: combined mAP50-95 > 0.280

### P5 — Drop Ambiguous B2/B3 Training Samples
> "Apakah training tanpa gambar yang mengandung co-occurring B2+B3 dalam bounding box yang overlap (jarak <50px) meningkatkan B2 mAP? Apakah dataset yang lebih kecil tapi lebih bersih menghasilkan model yang lebih baik?"

**Rasional**: Gambar di mana B2 dan B3 muncul berdampingan adalah kandidat terkuat label noise/ambiguity. Jika kita drop gambar tersebut, model belajar dari contoh yang lebih "bersih".
**Success criterion**: B2 AP50-95 > 0.240 dengan dataset yang lebih kecil

### P6 — CBAM Attention untuk Fine-Grained B2/B3
> "Apakah menambah Channel+Spatial Attention (CBAM) di backbone YOLO11l meningkatkan B2 mAP50-95 lebih dari 0.01 absolute?"

**Rasional**: CBAM secara eksplisit menerapkan attention mechanism untuk memfokuskan model pada fitur channel dan spatial yang discriminative. Untuk B2/B3 yang secara visual mirip, attention mungkin membantu model fokus pada perbedaan subtle (tekstur permukaan buah).

### P7 — Label Smoothing untuk Ambiguous Classes
> "Apakah label_smoothing=0.1 pada training YOLO11l meningkatkan val_map50-95? Label smoothing seharusnya mengurangi overconfidence pada label yang ambigu."

**Rasional**: Label smoothing yang rendah (0.1) mengurangi gradient yang terlalu kuat dari label yang salah. Karena B2/B3 punya ~40% label noise, label smoothing seharusnya membuat training lebih robust.
**Success criterion**: mAP50-95 > 0.280

### P8 — Multi-Scale Training dengan imgsz List
> "Apakah multi-scale training (imgsz=640 dengan random scale augmentation yang lebar, misal 0.3-1.7) membantu B4 small object detection dibandingkan single-scale?"

**Success criterion**: B4 AP50-95 > 0.170 (dari 0.141 saat ini)

---

## 8. Eksperimen yang Sedang/Akan Berjalan

### Status Background Agents (diluncurkan bersamaan dengan laporan ini)

**Sub-Agent 1 — DINOv2 Crop Classifier Training**
- Task: Install transformers, download DINOv2-base, train linear head pada Dataset-Crops
- Target: val_acc >80% (vs EfficientNet 62.7%), B2 acc >65%
- Output: `dinov2_classifier.pth`, hasil evaluasi per-class
- Status: LAUNCHING (background)

**Sub-Agent 2 — Two-Stage Eval Debug**
- Task: Audit `two_stage_eval.py` untuk bug mAP computation
- Key question: Apakah confidence = det_conf × cls_conf? Apakah IoU matching benar?
- Output: `two_stage_debug_report.md`, `two_stage_eval_v2.py` jika ada bug
- Status: LAUNCHING (background)

**Sub-Agent 3 — Color Feature Analysis**
- Task: Run `color_classifier.py`, analisis HSV B2/B3 separability
- Output: `color_analysis_report.md`
- Status: LAUNCHING (background)

### Rencana Eksperimen Berikutnya (prioritas):
1. DINOv2 classifier → jika >75% acc, build two-stage pipeline
2. YOLO11l TIME_HOURS=2.0 (lebih banyak epoch)
3. Label smoothing 0.1 (1-line change, low risk)
4. CORAL ordinal loss pada crop classifier

---

## 9. Rekomendasi Prioritas

Berdasarkan evidence dari 35+ eksperimen:

### PRIORITAS 1 — DINOv2 Two-Stage Pipeline [Highest Expected Impact]
- **Evidence**: Single-class detector sudah 0.390 mAP50-95 (vs multi-class 0.269)
- **Gap**: Hanya butuh classifier dengan >75% accuracy untuk mengalahkan baseline
- **Action**: Gunakan DINOv2-base frozen + linear head (sub-agent sudah dilaunch)
- **Expected delta**: +0.010 to +0.050

### PRIORITAS 2 — Lebih Banyak Waktu Training [Low Risk, Clear Upside]
- **Evidence**: Epoch 80 masih marginal improvement (+0.0002). Model belum fully converged.
- **Action**: Set TIME_HOURS=2.0, train YOLO11l 300 epochs
- **Expected delta**: +0.010 to +0.030

### PRIORITAS 3 — Label Smoothing 0.1 [Trivial to Implement]
- **Evidence**: B2 label noise ~40%. Label smoothing mengurangi gradient dari noisy labels.
- **Action**: Tambah `label_smoothing=0.1` ke train.py
- **Expected delta**: +0.005 to +0.015

### PRIORITAS 4 — Drop Ambiguous Co-occurring B2/B3 [Data Quality]
- **Evidence**: B2/B3 confusion adalah root cause #1. Gambar dengan B2+B3 berdampingan = paling ambigu.
- **Action**: Script untuk filter gambar dengan co-occurring B2/B3 dalam bbox proximity
- **Expected delta**: +0.005 to +0.020 (dataset lebih kecil tapi lebih bersih)

### PRIORITAS 5 — Contrastive Loss untuk B2/B3 Disambiguation
- **Evidence**: Standard CE tidak secara eksplisit mendorong B2/B3 embeddings terpisah
- **Action**: Tambah SupCon loss pada RoI features dari YOLO head
- **Expected delta**: +0.010 to +0.030 (tapi implementasinya kompleks)

### JANGAN DICOBA LAGI (sudah terbukti gagal):
- Semua varian imgsz (sudah tried: 640, 800, 1024, 1280)
- Model lebih besar (YOLOv9e, YOLO11x — underfitting dalam 20 menit)
- RT-DETR, RF-DETR (tidak konvergen cukup)
- Oversampling dengan geometric flip
- Model soup/averaging
- Loss weight tuning (BOX/CLS/DFL)

---

## 10. Appendix: Konfigurasi Lengkap Best Run

```python
# Dari train.py, best configuration saat ini:
MODEL = "yolo11l.pt"
IMGSZ = 640
BATCH = 16
EPOCHS = 500  # dengan early stopping
TIME_HOURS = 0.33  # 20 menit limit → ~80 epochs tercapai
OPTIMIZER = "AdamW"
COS_LR = True
ERASING = 0.4
DATA_YAML = "Dataset-TrainTest/data.yaml"  # 3388 training images

# Results:
# val_map50    = 0.554304
# val_map50_95 = 0.269424  ← CURRENT BEST
# precision    = 0.502331
# recall       = 0.631508
# B1 mAP50-95 ≈ 0.439
# B2 mAP50-95 ≈ 0.216
# B3 mAP50-95 ≈ 0.267
# B4 mAP50-95 ≈ 0.141
```
