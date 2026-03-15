# Proposal 5: Multi-View / Tree-Level Context

## Ringkasan

Dataset ini secara alami multi-view:
- mayoritas pohon punya 4 sisi
- sebagian pohon punya 8 sisi

Pipeline repo saat ini masih image-level atau crop-level. Formulasi ini mencoba memanfaatkan set gambar per pohon sebagai sumber context tambahan.

## Hal Penting

Ini bukan berarti semua buah di satu pohon harus dipaksa punya kelas sama.

Audit lokal menunjukkan banyak pohon mengandung campuran kelas. Jadi tree-level context harus dipakai sebagai:
- context embedding
- view consistency regularizer
- atau set-level prior lunak

bukan sebagai hard label override.

## Kenapa Ini Baru

Repo sebelumnya sudah mengisyaratkan ide agregasi multi-view, tetapi saya belum melihat branch final yang benar-benar dieksekusi end-to-end untuk tujuan ini.

Jadi ini masih termasuk formulasi baru yang masuk akal.

## Referensi Primer

- [Deep Sets, NeurIPS 2017](https://papers.nips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html)
- [Set Transformer, ICML 2019](https://proceedings.mlr.press/v97/lee19d.html)
- [General Multi-Label Image Classification With Transformers, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Lanchantin_General_Multi-Label_Image_Classification_With_Transformers_CVPR_2021_paper.html)

Relevansi:
- Deep Sets / Set Transformer untuk agregasi set multi-view
- C-Tran relevan untuk dependency antar label/kehadiran label pada satu gambar

## Hipotesis

Jika beberapa view dari pohon yang sama dipakai sebagai sumber context embedding, maka model bisa membedakan ambiguity visual yang tidak cukup terlihat dari satu crop saja, terutama untuk `B2/B3`.

## Kapan Proposal Ini Layak

Proposal ini lebih mahal secara desain daripada Proposal 1-3.

Jangan mulai dari sini jika:
- hierarchical branch belum dicoba
- specialist branch belum dibuat

Gunakan proposal ini jika:
- Anda ingin membuka jalur benar-benar baru
- Anda siap membangun dataset/index per pohon

## Batasan

- jangan menjadikan ID pohon sebagai shortcut label
- jangan memaksa voting mayoritas sebagai label akhir box
- gunakan set-level context secara lunak

## Bentuk Implementasi yang Disarankan

### Opsi A: Context Embedding Per Tree

Langkah:
- encode beberapa gambar satu pohon
- agregasikan dengan `Deep Sets` atau `Set Transformer`
- gunakan embedding itu sebagai context tambahan untuk classifier/crop branch

### Opsi B: Consistency Regularization Antar View

Langkah:
- view yang sama pohonnya didorong memiliki representasi environment yang konsisten
- tetapi prediksi box tetap image-specific

### Opsi C: Image-Level Partial Label Prior

Langkah:
- model belajar set label yang mungkin hadir di gambar/pohon
- box classifier memakai prior lunak itu, bukan hard override

## File Repo yang Perlu Dihubungkan

Perlu file baru hampir pasti.

Kemungkinan:
- `scripts/build_tree_view_index.py`
- `scripts/train_tree_context_encoder.py`
- `scripts/train_multiview_b23_specialist.py`

## Artefak yang Harus Dibuat

- index pohon -> daftar gambar/view
- statistik konsistensi kelas per pohon
- context encoder
- evaluasi apakah context membantu branch `B2/B3`

## Tahapan Eksekusi

### Tahap 0: Bangun index per pohon

Gunakan format nama file:
- contoh `DAMIMAS_A21B_0001_1.jpg`

Yang perlu dibangun:
- mapping `tree_id -> views`
- mapping split
- statistik campuran kelas per tree

### Tahap 1: Proof-of-concept context only

Jangan langsung bangun full multi-view pipeline besar.

Mulailah dari:
- satu encoder image/crop
- satu aggregator set
- satu tugas kecil: bantu specialist `B2/B3`

### Tahap 2: Ukur gain terhadap binary branch

Pertanyaan inti:
- apakah tambahan context per tree mengangkat binary `B2/B3`?

Kalau tidak, berhenti cepat.

### Tahap 3: Integrasi ke pipeline

Kalau binary gate lolos:
- baru integrasikan ke hierarchical pipeline

## Evaluasi Wajib

Laporkan:
- seberapa konsisten distribusi kelas per tree
- binary `B2/B3` accuracy dengan dan tanpa tree context
- end-to-end `mAP50-95` jika sudah diintegrasikan

## Kill Criterion

Hentikan branch ini jika:
- tree context ternyata tidak informatif
- model malah belajar shortcut ID pohon
- binary gate tidak naik
- kompleksitas implementasi jauh lebih besar dari gain awal

## Instruksi Singkat untuk Agent Pelaksana

Suruh agent:

1. mulai dari membangun index pohon dan statistik multi-view
2. jangan langsung mengasumsikan satu pohon = satu kelas
3. gunakan tree context sebagai prior lunak atau context embedding
4. validasi dulu pada specialist `B2/B3`, baru ke pipeline penuh

## Prompt Siap Pakai untuk Agent

```text
Baca file research/formulasi-baru-30pct/05-multiview-tree-context.md. Mulai dari membangun index multi-view per pohon dan statistik konsistensi kelas. Lalu buat proof-of-concept context encoder berbasis Deep Sets atau Set Transformer untuk membantu specialist B2/B3. Jangan gunakan tree ID sebagai shortcut label keras. Integrasikan ke pipeline hanya jika binary gate benar-benar membaik.
```
