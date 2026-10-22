\# UTS Graf Sains Data



Nama: Alfathrindra Agastyo Yulianto

NIM: 23031554094

Kelas: 2023 B\\





\## Deskripsi Proyek



Proyek Ujian Tengah Semester (UTS) mata kuliah Graf Sains Data ini mengolah dataset interaksi e-commerce RetailRocket untuk membangun dan menganalisis graf menggunakan Neo4j, Python, dan pendekatan \*Vector Space Model\* (VSM).



Analisis mencakup pemisahan data berdasarkan waktu, pembentukan graf interaksi pengunjung dan barang, proyeksi graf User–Item menjadi User–User dan Item–Item, perhitungan kemiripan menggunakan cosine similarity, serta rekomendasi barang menggunakan K-Nearest Neighbors (KNN).



\## Tujuan



1\. Melakukan \*data wrangling\* dan eksplorasi data berdasarkan periode training dan testing.

2\. Membentuk serta menganalisis graf interaksi Visitor–Item menggunakan Neo4j.

3\. Membandingkan proyeksi User–User dan Item–Item berdasarkan ukuran matriks, sparsity, kebutuhan memori, dan waktu pemrosesan.

4\. Menghitung kemiripan antar-Item menggunakan cosine similarity dan menyimpan relasi kemiripan ke Neo4j.

5\. Menguji rekomendasi Item menggunakan KNN serta mengevaluasinya dengan Hit\\@5 dan NDCG\\@5.



\## Teknologi



\- Python

\- Neo4j dan Cypher

\- Neo4j Python Driver

\- Pandas dan NumPy

\- SciPy Sparse Matrix

\- NetworkX

\- Matplotlib

\- Scikit-learn

\- Jupyter Notebook



\## Struktur Proyek



```text

UTS/

├── data/

│   ├── raw/

│   │   └── events.csv

│   └── processed/

│       ├── events\_training.csv

│       └── events\_testing.csv

├── outputs/

│   └── Hasil analisis, visualisasi, dan evaluasi

├── 1\_2\_neo4j\_driver.py

├── filter\_events.py

├── composition\_graph.py

├── import\_retailrocket.py

├── vsm\_users.py

├── vsm\_items.py

├── calculate\_similarity.py

├── write\_similarity\_to\_neo4j.py

├── analytics\_knn.py

├── evaluate\_knn\_sample.py

├── requirements.txt

├── .gitignore

└── README.md

```



Notebook `.ipynb` proyek juga disertakan dalam repository.



\## Dataset



Dataset yang digunakan adalah RetailRocket eCommerce Dataset, yang berisi catatan interaksi pengunjung terhadap barang.



Sumber dataset: \[RetailRocket eCommerce Dataset — Kaggle](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)



File utama yang digunakan dalam proses pemfilteran adalah `data/raw/events.csv`. File training dan testing disimpan dalam `data/processed/`.



\## Periode Data



\- \*\*Training:\*\* 17 Mei 2015–17 Juli 2015.

\- \*\*Testing:\*\* 26 Juli 2015–10 Agustus 2015.



Periode testing dipisahkan dari training berdasarkan timestamp. Data interaksi pada periode training digunakan untuk membangun graf referensi, sedangkan data testing digunakan untuk evaluasi.



\## Menyiapkan Lingkungan



Pastikan Python dan Neo4j Desktop sudah terpasang.



Buat virtual environment menggunakan PowerShell:



```powershell

python -m venv myvenv

.\\myvenv\\Scripts\\Activate.ps1

```



Instal dependensi:



```powershell

python -m pip install --upgrade pip

pip install -r requirements.txt

```



Jalankan Neo4j dan pastikan konfigurasi koneksi sesuai dengan database lokal yang digunakan. Konfigurasi URI, username, dan password harus disesuaikan dengan lingkungan masing-masing.



\## Alur Analisis



1\. Memfilter dataset berdasarkan timestamp dan menyimpan CSV training/testing.

2\. Mengimpor interaksi training ke Neo4j sebagai graf Visitor–Item.

3\. Menghitung komposisi node dan relationship graf.

4\. Membentuk representasi sparse User–Item serta proyeksi User–User dan Item–Item.

5\. Membandingkan karakteristik dan efisiensi kedua proyeksi.

6\. Menghitung cosine similarity antar-Item.

7\. Menyimpan relasi `SIMILAR\_TO` beserta nilai similarity ke Neo4j.

8\. Memvisualisasikan graf sebelum dan sesudah penambahan relasi similarity.

9\. Menjalankan evaluasi rekomendasi KNN menggunakan Hit\\@5 dan NDCG\\@5.



\## Hasil Graf Training



Graf training yang dihasilkan memiliki komposisi berikut:



| Komponen                        |    Jumlah |

| ------------------------------- | --------: |

| Visitor nodes                   |   668.072 |

| Item nodes                      |   170.631 |

| `VIEWED`                        | 1.259.024 |

| `ADDED\_TO\_CART`                 |    31.691 |

| `BOUGHT`                        |    10.486 |

| Total interaction relationships | 1.301.201 |



Jumlah tersebut mengacu pada hasil pemrosesan data training yang digunakan dalam eksperimen.



\## Catatan



\- Matriks VSM menggunakan representasi sparse untuk mengurangi kebutuhan memori.

\- Relasi kemiripan Item dihitung dari pola interaksi, bukan dari atribut harga.

\- Hasil evaluasi rekomendasi bergantung pada data training, data testing, parameter KNN, dan pengguna yang dievaluasi.

\- File lingkungan virtual, cache, ZIP dataset, serta file data yang tidak digunakan dalam pipeline dikecualikan melalui `.gitignore`.
