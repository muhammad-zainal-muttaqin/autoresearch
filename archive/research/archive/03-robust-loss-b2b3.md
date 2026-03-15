# Proposal 3: Robust Loss Khusus untuk Branch B2/B3

## Ringkasan

Formulasi ini berangkat dari asumsi berikut:

- `B2` dan `B3` mungkin bukan sekadar hard classes
- boundary-nya mungkin juga mengandung ambiguity/noisy label effect

Alih-alih mencoba mengoreksi label, branch ini mengubah loss agar lebih tahan terhadap ambiguity tersebut.

## Kenapa Ini Baru

Repo sudah mencoba:
- label correction berbasis model disagreement
- CE biasa
- ordinal/CORN

Repo belum benar-benar mencoba:
- robust loss khusus noisy labels pada specialist branch
- penggunaan loss robust hanya pada bagian masalah yang paling ambiguity-heavy

## Referensi Primer

- [Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels, NeurIPS 2018](https://proceedings.neurips.cc/paper/2018/hash/f2925f97bc13ad2852a7a551802feea0-Abstract.html)
- [Symmetric Cross Entropy for Robust Learning With Noisy Labels, ICCV 2019](https://openaccess.thecvf.com/content_ICCV_2019/html/Wang_Symmetric_Cross_Entropy_for_Robust_Learning_With_Noisy_Labels_ICCV_2019_paper.html)
- [Fine-Grained Classification With Noisy Labels, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Wei_Fine-Grained_Classification_With_Noisy_Labels_CVPR_2023_paper.html)

## Hipotesis

Jika ambiguity `B2/B3` memang bertindak seperti localized noisy labels, maka robust loss pada specialist branch akan menahan memorization terhadap boundary yang salah/ambigu, sehingga binary `B2/B3` menjadi lebih stabil.

## Posisi Proposal Ini

Ini bukan branch utama pertama.

Branch ini paling masuk akal sebagai:
- upgrade untuk Proposal 2
- atau fallback ringan jika agent belum siap mengimplementasikan contrastive learning penuh

## Batasan

- jangan pasang robust loss ke semua branch sekaligus di eksperimen pertama
- jangan pakai branch ini untuk membenarkan pengulangan one-stage tuning
- fokus ke specialist `B2/B3`, bukan 4-way full classifier

## Implementasi yang Disarankan

Urutan implementasi:

1. `GCE`
2. `SCE`
3. jika perlu, hybrid `CE + robust loss`

Mulailah dari yang paling sederhana.

## File Repo yang Perlu Dihubungkan

- `scripts/stage2_models.py`
- `scripts/train_dinov2_imagefolder_classifier.py`
- file baru yang dedicated untuk binary `B2/B3` robust-loss training

Kemungkinan file baru:
- `scripts/train_b23_gce.py`
- `scripts/train_b23_sce.py`

## Artefak yang Harus Dibuat

- trainer binary `B2/B3` dengan GCE atau SCE
- evaluasi binary lengkap
- perbandingan langsung terhadap baseline binary sebelumnya

## Tahapan Eksekusi

### Tahap 0: Tetapkan baseline binary

Sebelum mencoba robust loss, agent harus punya baseline binary `B2/B3` yang fair:
- dataset sama
- crop policy sama
- backbone sama

Kalau baseline belum ada, buat dulu.

### Tahap 1: Implement GCE

Task:
- ganti CE dengan GCE pada binary branch
- jaga semua hal lain tetap konstan

Tujuan:
- mengisolasi efek objective

### Tahap 2: Jika perlu, implement SCE

Task:
- bandingkan GCE vs SCE
- jangan campur dengan perubahan arsitektur lain

### Tahap 3: Integrasi

Hanya jika binary gate membaik:
- sambungkan ke hierarchical evaluator

## Evaluasi Wajib

Laporkan:
- binary accuracy
- precision/recall per kelas
- confusion matrix
- training stability
- end-to-end `mAP50-95` jika diintegrasikan

## Kill Criterion

Hentikan branch ini jika:
- robust loss tidak mengangkat binary `B2/B3`
- hasil hanya memindahkan error dari `B2` ke `B3`
- accuracy naik tipis tapi pipeline tidak berubah

## Instruksi Singkat untuk Agent Pelaksana

Suruh agent:

1. buat baseline binary yang adil dulu
2. ubah hanya objective, bukan sekaligus backbone + augmentasi + evaluator
3. bandingkan GCE dan SCE satu per satu
4. integrasikan ke hierarchical pipeline hanya bila gate binary membaik nyata

## Prompt Siap Pakai untuk Agent

```text
Baca file research/formulasi-baru-30pct/03-robust-loss-b2b3.md. Fokus hanya pada specialist branch B2/B3. Implementasikan robust loss untuk noisy/ambiguous labels, mulai dari GCE lalu SCE jika perlu. Jangan mengubah banyak faktor sekaligus. Ukur binary gate secara fair, lalu integrasikan ke hierarchical pipeline hanya jika hasilnya benar-benar lebih baik.
```
