# Proposal 1: Hierarchical + Context-Aware Two-Stage

## Ringkasan

Formulasi ini memecah klasifikasi 4 kelas menjadi dua keputusan:

1. `B1 / B23 / B4`
2. jika `B23`, baru lanjut `B2 / B3`

Perubahan kuncinya bukan hanya hierarki, tetapi juga konteks yang lebih lebar. Stage-2 saat ini terlalu sempit karena crop classifier dilatih dari crop `PAD_RATIO=0.2`, padahal `B2` dan `B3` kemungkinan membutuhkan konteks sekitar tandan, pelepah, dan pola lingkungan lokal.

## Kenapa Ini Masih Layak

Branch yang sudah gagal di repo:
- flat 4-way EfficientNet
- flat 4-way DINOv2 CE
- flat 4-way DINOv2 CORN

Yang belum benar-benar dieksekusi end-to-end:
- hierarki `B1/B23/B4 -> B2/B3`
- context-aware stage-2 dengan crop lebar

Jadi ini bukan tuning ulang. Ini perubahan struktur keputusan.

## Referensi Primer

- [Hierarchical Fine-Grained Image Forgery Detection and Localization, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Guo_Hierarchical_Fine-Grained_Image_Forgery_Detection_and_Localization_CVPR_2023_paper.html)
- [From Coarse to Fine-Grained Open-Set Recognition, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Lang_From_Coarse_to_Fine-Grained_Open-Set_Recognition_CVPR_2024_paper.html)

Catatan: kedua paper ini bukan domain sawit, tetapi relevan untuk ide coarse-to-fine dan hierarchical dependency.

## Hipotesis

Jika stage-2 dipecah menjadi coarse classifier dan specialist binary classifier, lalu classifier diberi konteks crop yang lebih lebar, maka `B2/B3` akan lebih mudah dipisahkan tanpa merusak `B1` dan `B4`, sehingga two-stage pipeline bisa menutup gap besar antara single-class detector `0.390` dan best 4-class one-stage `0.269`.

## Tujuan Eksperimen

Target minimal:
- hierarchical two-stage `mAP50-95 > 0.269424`

Target kuat:
- `mAP50-95 >= 0.30`

Target diagnostik:
- `B2 AP50-95` two-stage naik jelas di atas `~0.10`

## Batasan

- jangan sentuh branch one-stage
- jangan ulang flat 4-way classifier
- jangan pakai `PAD_RATIO=0.2` sebagai satu-satunya setup
- eksperimen pertama harus fokus ke validasi formulasi, bukan ke sweep hyperparameter besar

## File Repo yang Relevan

- `scripts/make_single_class_dataset.py`
- `scripts/make_crop_dataset.py`
- `scripts/make_hierarchical_crop_datasets.py`
- `scripts/train_dinov2_imagefolder_classifier.py`
- `scripts/two_stage_eval_v2.py`
- `scripts/two_stage_hierarchical_eval.py`
- `scripts/stage2_models.py`

## Artefak yang Harus Dibuat

Minimal:
- dataset crop konteks lebar untuk coarse branch
- dataset crop konteks lebar untuk binary `B2/B3`
- checkpoint coarse classifier
- checkpoint binary classifier
- evaluator hierarchical yang bisa memilih `pad_ratio`
- laporan eksperimen di `research/experiment-journal.md`

Sebaiknya:
- satu script baru untuk membangun crop dataset dengan `pad_ratio` yang bisa diatur
- satu script training yang generik untuk dataset `ImageFolder`

## Tahapan Eksekusi

### Tahap 0: Validasi asumsi

Agent harus memastikan:
- single-class detector checkpoint masih tersedia
- `Dataset-Crops` bisa dibuat ulang
- evaluator hierarchical bisa dijalankan lokal

Jika ada aset yang hilang, hentikan dulu dan catat blocker.

### Tahap 1: Ubah formulasi data crop

Jangan memakai hanya crop sempit 20%.

Buat ulang dataset crop dengan setidaknya dua konteks:
- `pad_ratio=0.2`
- `pad_ratio=0.6` atau `0.8`

Tujuan:
- membandingkan apakah `B2/B3` memang membutuhkan konteks lebih besar

Output yang diharapkan:
- dataset coarse3 dengan crop lebar
- dataset binary `B2/B3` dengan crop lebar

### Tahap 2: Train coarse classifier

Task:
- train classifier 3 kelas: `B1`, `B23`, `B4`

Ekspektasi:
- ini harus jauh lebih mudah dari 4-way classifier
- jika classifier ini lemah, branch hierarchical hampir pasti gagal

Gate awal:
- coarse val accuracy harus tinggi dan stabil
- target praktis: `>= 80%`

### Tahap 3: Train binary specialist `B2/B3`

Task:
- train classifier khusus `B2` vs `B3`
- gunakan crop lebar yang sama

Gate awal:
- binary accuracy harus melampaui baseline implicit dari flat classifier
- target praktis awal: `>= 70%`
- target kuat: `>= 75%`

### Tahap 4: End-to-end hierarchical eval

Gunakan:
- single-class detector existing
- coarse classifier
- binary classifier
- evaluator hierarchical

Harus diuji minimal di dua setting:
- `pad_ratio` sempit
- `pad_ratio` lebar

### Tahap 5: Analisis

Agent harus menjawab:
- apakah coarse branch memang mudah?
- apakah specialist `B2/B3` naik signifikan?
- apakah peningkatan classifier benar-benar terkonversi ke `mAP50-95`?
- apakah konteks lebar membantu atau malah menambah noise?

## Evaluasi Wajib

Laporkan:
- coarse classifier accuracy
- binary `B2/B3` accuracy
- end-to-end `mAP50`
- end-to-end `mAP50-95`
- AP per kelas
- delta terhadap baseline one-stage `0.269424`

## Kill Criterion

Hentikan branch ini jika salah satu terjadi:

- coarse classifier gagal kuat, mis. tidak bisa stabil di kisaran tinggi
- binary `B2/B3` tetap dekat chance / tidak naik berarti
- end-to-end tetap jauh di bawah best one-stage walau classifier internal membaik kecil

Praktis:
- jika setelah satu siklus penuh hasil end-to-end masih `< 0.24`, jangan sweep lama
- kalau coarse bagus tapi binary gagal, pindah ke Proposal 2

## Instruksi Singkat untuk Agent Pelaksana

Suruh agent:

1. jangan sentuh one-stage branch
2. buat ulang crop dataset dengan context window yang bisa diatur
3. latih dua classifier terpisah: coarse3 dan binary `B2/B3`
4. evaluasi hanya dengan evaluator hierarchical
5. bandingkan `pad_ratio=0.2` vs `pad_ratio>=0.6`
6. putuskan cepat apakah branch ini hidup atau mati

## Prompt Siap Pakai untuk Agent

```text
Baca file research/formulasi-baru-30pct/01-hierarchical-context-aware-two-stage.md dan ikuti persis. Jangan ulang one-stage tuning. Fokus hanya pada branch hierarchical two-stage. Buat dataset crop dengan context window yang bisa diatur, latih classifier coarse3 dan classifier binary B2/B3, lalu evaluasi end-to-end dengan evaluator hierarchical. Laporkan classifier gate, AP per kelas, mAP50-95, dan keputusan keep/discard.
```
