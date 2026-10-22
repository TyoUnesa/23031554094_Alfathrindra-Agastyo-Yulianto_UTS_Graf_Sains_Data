# UTS Graf Sains Data

**Nama:** Alfathrindra Agastyo Yulianto  
**NIM:** 23031554094  
**Kelas:** 2023 B

## Deskripsi Proyek

Proyek Ujian Tengah Semester (UTS) mata kuliah Graf Sains Data ini menggunakan dataset RetailRocket untuk membangun dan menganalisis graf interaksi e-commerce menggunakan Python dan Neo4j.

Analisis mencakup pemisahan dataset berdasarkan waktu, pembentukan graf Visitor–Item, representasi *Vector Space Model* (VSM), perbandingan proyeksi graf User–User dan Item–Item, perhitungan *cosine similarity*, serta rekomendasi barang menggunakan K-Nearest Neighbors (KNN).

## Tujuan

1. Melakukan *data wrangling* dan eksplorasi data training dan testing.
2. Membangun dan menganalisis graf interaksi Visitor–Item di Neo4j.
3. Membandingkan proyeksi User–User dan Item–Item berdasarkan dimensi, sparsity, memori, dan waktu pemrosesan.
4. Menghitung kemiripan antar-Item dan menyimpan relasi `SIMILAR_TO` di Neo4j.
5. Mengevaluasi rekomendasi menggunakan metrik Hit@5 dan NDCG@5.

## Teknologi

- Python
- Neo4j dan Cypher
- Pandas dan NumPy
- SciPy Sparse Matrix
- NetworkX
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## Struktur Proyek

```text
UTS/
├── data/
│   ├── raw/
│   │   └── events.csv
│   └── processed/
│       ├── events_training.csv
│       └── events_testing.csv
├── outputs/
│   ├── graph_before_similarity.png
│   ├── graph_after_similarity.png
│   ├── user_item_vsm.npz
│   ├── item_user_vsm.npz
│   ├── user_projection.png
│   ├── item_projection.png
│   └── hasil analisis lainnya
├── 1_2_neo4j_driver.py
├── filter_events.py
├── composition_graph.py
├── import_retailrocket.py
├── vsm_users.py
├── vsm_items.py
├── calculate_similarity.py
├── write_similarity_to_neo4j.py
├── analytics_knn.py
├── evaluate_knn_sample.py
├── requirements.txt
├── .gitignore
├── README.md
└── notebook UTS Graf Sains Data
```

## Dataset

Dataset yang digunakan adalah RetailRocket eCommerce Dataset.

Sumber: [RetailRocket eCommerce Dataset di Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)

File utama yang digunakan dalam proses pengolahan adalah `data/raw/events.csv`. Hasil pemfilteran disimpan dalam folder `data/processed/`.

## Periode Data

| Pembagian | Periode |
|---|---|
| Training | 17 Mei–17 Juli 2015 |
| Testing | 26 Juli–10 Agustus 2015 |

Pemisahan dilakukan berdasarkan timestamp. Data training digunakan untuk membangun graf referensi, sedangkan data testing digunakan untuk evaluasi rekomendasi.

## Hasil Graf Training

Komposisi graf yang diperoleh dari proses import data training ke Neo4j adalah sebagai berikut.

| Komponen | Jumlah |
|---|---:|
| Visitor | 668.072 |
| Item | 170.631 |
| `VIEWED` | 1.259.024 |
| `ADDED_TO_CART` | 31.691 |
| `BOUGHT` | 10.486 |
| Total relationship interaksi | 1.301.201 |

## Alur Analisis

1. Memfilter dataset berdasarkan timestamp dan menyimpan data training/testing.
2. Mengimpor interaksi training ke Neo4j.
3. Menghitung komposisi node dan relationship.
4. Membentuk matriks VSM User–Item dan Item–User.
5. Membandingkan proyeksi User–User dan Item–Item.
6. Menghitung *cosine similarity* antar-Item.
7. Menyimpan relasi `SIMILAR_TO` beserta nilai similarity ke Neo4j.
8. Membandingkan visualisasi graf sebelum dan sesudah penambahan relasi similarity.
9. Menjalankan KNN dan mengevaluasi rekomendasi menggunakan Hit@5 dan NDCG@5.

## Menyiapkan Lingkungan

Pastikan Python, Jupyter Notebook, dan Neo4j Desktop sudah terpasang.

Buat dan aktifkan virtual environment menggunakan PowerShell:

```powershell
python -m venv myvenv
.\myvenv\Scripts\Activate.ps1
```

Instal dependensi:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jalankan database Neo4j dan sesuaikan konfigurasi koneksi dengan lingkungan lokal. Masukkan kredensial sesuai permintaan script; jangan menyimpan password atau token pribadi di repository.

## Catatan

- Matriks VSM menggunakan representasi sparse untuk menghemat memori.
- Kemiripan Item dihitung berdasarkan pola interaksi pengunjung.
- Hasil evaluasi rekomendasi bergantung pada data, parameter, dan pengguna yang dievaluasi.
- Folder virtual environment, arsip dataset, file data yang tidak digunakan, dan cache dikecualikan menggunakan `.gitignore`.

## Penulis

**Alfathrindra Agastyo Yulianto**  
Program Studi Sains Data  
Universitas Negeri Surabaya
