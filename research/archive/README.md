# Formulasi Baru Menuju 0.30+ mAP50-95

Folder ini berisi proposal formulasi yang sengaja dibatasi hanya pada arah yang benar-benar berbeda dari history repo saat ini.

Tujuan folder ini:
- menghindari pengulangan branch yang sudah gagal
- memberi brief yang langsung bisa dieksekusi agent lain
- menjaga eksperimen tetap ilmiah: ada hipotesis, deliverable, evaluasi, dan kill criterion

## State Repo Saat Dokumen Ini Dibuat

- Best one-stage 4 kelas: `0.269424`
- Best single-class detector: `0.390430`
- Best flat two-stage yang sudah dievaluasi benar: `0.181088`
- Long-run one-stage `TIME_HOURS=2.0` sudah gagal
- Flat crop classification dengan EfficientNet, DINOv2 CE, dan DINOv2 CORN sudah gagal

## Yang Tidak Boleh Diulang

Jangan buka branch baru yang pada dasarnya hanya mengulang hal-hal berikut:

- one-stage tuning biasa pada `train.py`
- frozen-backbone crop classifier 4-way dengan cross-entropy biasa
- ordinal/CORN 4-way pada crop classifier
- copy-paste / LR / batch / epoch sweep tanpa perubahan formulasi
- detector baru tanpa mengubah masalah klasifikasi

## Urutan Prioritas

1. [01-hierarchical-context-aware-two-stage.md](./01-hierarchical-context-aware-two-stage.md)
2. [02-b2b3-contrastive-prototype.md](./02-b2b3-contrastive-prototype.md)
3. [03-robust-loss-b2b3.md](./03-robust-loss-b2b3.md)
4. [04-part-selection-fgvc-stage2.md](./04-part-selection-fgvc-stage2.md)
5. [05-multiview-tree-context.md](./05-multiview-tree-context.md)

## Cara Pakai Folder Ini

Kalau Anda akan menyuruh agent lain mengeksekusi:

1. Pilih satu file proposal saja.
2. Suruh agent membaca proposal itu sampai habis sebelum menulis kode.
3. Suruh agent mematuhi bagian `Batasan`, `Tahapan`, `Evaluasi`, dan `Kill Criterion`.
4. Jangan gabungkan dua formulasi baru dalam satu eksperimen pertama.
5. Pastikan agent menulis hasilnya kembali ke `research/experiment-journal.md` dan `results.tsv` bila eksperimen benar-benar dijalankan.

## Saran Eksekusi Nyata

Kalau hanya boleh memilih satu branch terlebih dahulu, pilih:

- `01-hierarchical-context-aware-two-stage.md`

Alasan:
- paling dekat dengan failure mode repo saat ini
- tidak mengulang branch lama
- masih memanfaatkan aset terbaik repo: single-class detector `0.390`
- peluang naik ke `0.30` paling realistis dibanding branch lain
