# Proposal 4: Part-Selection / Fine-Grained Transformer untuk Stage-2

## Ringkasan

Formulasi ini mengasumsikan masalah utama stage-2 bukan hanya objective, tetapi juga cara model mencari bukti visual. Crop classifier yang dipakai sejauh ini terlalu generik: ia mengonsumsi crop dan langsung menghasilkan kelas.

Branch ini mengganti stage-2 menjadi model fine-grained yang secara eksplisit mencari patch atau region diskriminatif.

## Kenapa Ini Baru

Repo sudah mencoba backbone lebih kuat, tetapi masih dalam pola:
- frozen encoder
- shallow head
- flat logits

Repo belum benar-benar mencoba:
- fine-grained transformer yang secara eksplisit memilih region atau patch penting
- class-specific attention yang bisa menunjukkan bukti visual `B2` vs `B3`

## Referensi Primer

- [TransFG: A Transformer Architecture for Fine-grained Recognition](https://arxiv.org/abs/2103.07976)
- [A Simple Interpretable Transformer for Fine-Grained Image Classification and Analysis (INTR), ICLR 2024](https://arxiv.org/abs/2311.04157)
- [SIM-Trans: Structure Information Modeling Transformer for Fine-grained Visual Categorization](https://arxiv.org/abs/2208.14607)

## Hipotesis

Jika model stage-2 dipaksa untuk mengalokasikan perhatian ke patch/atribut yang benar-benar diskriminatif, maka branch `B2/B3` akan membaik karena model tidak lagi mengandalkan global crop appearance yang terlalu ambigu.

## Kapan Proposal Ini Dipakai

Jangan pakai ini sebagai branch pertama.

Ini cocok bila:
- hierarchical pipeline sudah ada
- binary `B2/B3` tetap sulit
- contrastive/prototype belum cukup

## Batasan

- jangan langsung implement arsitektur sangat besar dan sulit dipelihara
- pilih satu model FGVC yang paling ringan untuk proof-of-concept
- fokus awal tetap pada stage-2 binary atau coarse branch, bukan full pipeline redesign

## Implementasi yang Disarankan

Urutan paling pragmatis:

1. ambil satu ViT backbone yang ringan
2. tambahkan mekanisme attention/part selection
3. gunakan hanya untuk binary `B2/B3` dulu

Jangan mulai dari:
- model sangat besar
- pipeline multi-loss kompleks
- multi-branch besar yang sulit di-debug

## File Repo yang Perlu Dihubungkan

Belum ada file langsung yang siap.

Kemungkinan file baru:
- `scripts/train_b23_fgvc_transformer.py`
- `scripts/eval_b23_fgvc_transformer.py`

Integrasi nanti ke:
- `scripts/two_stage_hierarchical_eval.py`

## Artefak yang Harus Dibuat

- trainer stage-2 FGVC
- visualisasi attention map
- evaluasi binary `B2/B3`
- checkpoint yang bisa diintegrasikan ke hierarchical pipeline

## Tahapan Eksekusi

### Tahap 0: Proof-of-concept kecil

Agent jangan mencoba mengejar SOTA penuh.

Yang dibutuhkan:
- bukti bahwa attention model benar-benar fokus ke patch yang berbeda
- binary gate yang lebih baik dari branch lama

### Tahap 1: Train binary `B2/B3`

Mulailah dari binary saja.

Reason:
- paling jelas mengukur apakah part-selection membantu masalah inti

### Tahap 2: Visual audit

Wajib:
- simpan attention maps atau region weights
- cek apakah model menatap area tandan yang masuk akal

Kalau attention menyebar acak, branch ini lemah.

### Tahap 3: Integrasi

Hanya jika binary gate lolos:
- sambungkan ke hierarchical pipeline

## Evaluasi Wajib

Laporkan:
- binary accuracy
- confusion matrix
- contoh attention maps
- kualitas interpretabilitas
- end-to-end `mAP50-95` jika sudah diintegrasikan

## Kill Criterion

Hentikan branch ini jika:
- training berat tetapi binary gate tidak naik
- attention map tidak memberi bukti diskriminatif yang bermakna
- integrasi pipeline tidak mengangkat `B2`

## Instruksi Singkat untuk Agent Pelaksana

Suruh agent:

1. jangan langsung build arsitektur besar
2. buat proof-of-concept binary `B2/B3` dulu
3. wajib log attention maps
4. nilai branch ini dari dua sisi: metric dan interpretability

## Prompt Siap Pakai untuk Agent

```text
Baca file research/formulasi-baru-30pct/04-part-selection-fgvc-stage2.md. Jalankan proof-of-concept stage-2 binary B2/B3 berbasis fine-grained transformer atau part-selection model. Fokus pada bukti bahwa model menemukan region diskriminatif, bukan hanya menaikkan akurasi tipis. Integrasikan ke hierarchical pipeline hanya jika binary gate dan visual audit sama-sama meyakinkan.
```
