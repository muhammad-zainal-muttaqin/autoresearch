# Proposal 2: B2/B3 Specialist dengan Contrastive + Prototype Learning

## Ringkasan

Formulasi ini memperlakukan `B2` vs `B3` sebagai masalah embedding, bukan sekadar klasifikasi cross-entropy.

Tujuannya:
- membuat representasi `B2` dan `B3` lebih terpisah
- menahan efek ambiguity / noisy boundary
- memberi specialist branch yang benar-benar berbeda dari DINOv2 + CE dan DINOv2 + CORN

## Kenapa Ini Baru

Repo sudah mencoba:
- EfficientNet classifier
- DINOv2 frozen backbone + CE
- DINOv2 frozen backbone + CORN

Repo belum benar-benar mencoba:
- supervised contrastive learning khusus `B2/B3`
- prototype / metric-based classification untuk `B2/B3`
- objective yang memaksa struktur embedding, bukan hanya logits

## Referensi Primer

- [Supervised Contrastive Learning, NeurIPS 2020](https://openreview.net/forum?id=eyL3mJZPXZ6)
- [On Learning Contrastive Representations for Learning With Noisy Labels, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yi_On_Learning_Contrastive_Representations_for_Learning_With_Noisy_Labels_CVPR_2022_paper.html)
- [Fine-Grained Classification With Noisy Labels, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Fine-Grained_Classification_With_Noisy_Labels_CVPR_2023_paper.html)

## Hipotesis

Jika `B2/B3` dipelajari sebagai embedding problem dengan objective contrastive yang tahan ambiguity, maka decision boundary akan lebih stabil daripada CE biasa, sehingga specialist `B2/B3` bisa menjadi komponen kuat di hierarchical pipeline.

## Posisi Proposal Ini

Proposal ini paling ideal sebagai kelanjutan dari Proposal 1.

Urutan yang paling sehat:
- bangun hierarchical framework dulu
- kalau binary `B2/B3` masih lemah, ganti objective binary branch ke contrastive/prototype

## Batasan

- jangan langsung dipakai untuk 4-way full classifier
- fokus awal hanya pada binary `B2/B3`
- jangan pakai frozen backbone + linear head yang sama persis seperti branch lama

## File Repo yang Perlu Dihubungkan

- `scripts/stage2_models.py`
- `scripts/train_dinov2_imagefolder_classifier.py`
- `scripts/make_hierarchical_crop_datasets.py`
- `scripts/two_stage_hierarchical_eval.py`

Kemungkinan perlu file baru:
- `scripts/train_b23_contrastive.py`
- `scripts/eval_b23_embedding.py`

## Artefak yang Harus Dibuat

- model binary `B2/B3` berbasis embedding
- evaluasi binary: accuracy, confusion matrix, embedding visualization sederhana
- checkpoint yang bisa dipanggil evaluator hierarchical

Opsional tapi bagus:
- t-SNE atau UMAP embedding `B2/B3`
- prototype bank per kelas

## Formulasi yang Disarankan

Pilih salah satu dari dua bentuk awal:

### Opsi A: CE + Supervised Contrastive

Loss total:
- `L = L_ce + lambda * L_supcon`

Keuntungan:
- paling mudah diintegrasikan
- masih memberi probabilitas kelas eksplisit

### Opsi B: Prototype-Based Binary Classifier

Langkah:
- backbone menghasilkan embedding
- setiap kelas punya prototype / class center
- prediksi berdasarkan jarak ke prototype

Keuntungan:
- cocok untuk boundary yang kabur
- mudah dianalisis apakah `B2` dan `B3` benar-benar overlap

## Tahapan Eksekusi

### Tahap 0: Tegaskan scope

Agent harus menahan diri:
- jangan train 4-way contrastive dulu
- jangan ganti semua pipeline sekaligus

Fokus hanya:
- binary specialist `B2/B3`

### Tahap 1: Siapkan dataset binary

Gunakan:
- `Dataset-Crops-B23`

Kalau Proposal 1 juga dikerjakan:
- pakai versi crop lebar

### Tahap 2: Implement objective baru

Minimal:
- backbone encoder
- projection head / embedding head
- classifier head atau prototype head
- loss contrastive

Kalau ingin cepat:
- mulai dari CE + SupCon

### Tahap 3: Gate classifier

Laporkan:
- val accuracy binary
- confusion matrix
- jarak intra-class vs inter-class di embedding

Target awal:
- binary accuracy harus mengalahkan branch lama dengan margin jelas

### Tahap 4: Integrasi ke hierarchical pipeline

Kalau binary gate lolos:
- plug specialist ke evaluator hierarchical
- ukur dampak ke end-to-end `mAP50-95`

## Evaluasi Wajib

Laporkan:
- binary `B2/B3` accuracy
- precision/recall per class
- confusion matrix
- optional embedding visualization
- end-to-end `mAP50-95` jika sudah diintegrasikan

## Kill Criterion

Hentikan branch ini jika:
- binary accuracy tidak naik jelas dari baseline specialist sebelumnya
- embedding tetap menunjukkan overlap berat
- integrasi ke pipeline tidak memberi dampak ke `B2 AP50-95`

Praktis:
- kalau binary branch tetap mentok di kisaran lama, jangan sweep banyak backbone

## Instruksi Singkat untuk Agent Pelaksana

Suruh agent:

1. fokus hanya pada binary `B2/B3`
2. implement `CE + SupCon` terlebih dahulu, baru prototype jika perlu
3. ukur kualitas embedding, bukan cuma accuracy
4. integrasikan ke hierarchical pipeline hanya jika binary gate lolos

## Prompt Siap Pakai untuk Agent

```text
Baca file research/formulasi-baru-30pct/02-b2b3-contrastive-prototype.md dan jalankan hanya branch specialist B2/B3 berbasis embedding. Jangan mengulang DINOv2 CE atau CORN. Mulai dari binary classifier dengan CE + supervised contrastive loss, evaluasi embedding dan confusion matrix, lalu integrasikan ke hierarchical pipeline hanya jika binary gate benar-benar lebih baik.
```
