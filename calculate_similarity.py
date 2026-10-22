import os

import numpy as np
import pandas as pd
from scipy.sparse import load_npz


OUTPUT_DIR = "outputs"

MATRIX_PATH = os.path.join(
    OUTPUT_DIR, "item_user_vsm.npz"
)
ITEM_INDEX_PATH = os.path.join(
    OUTPUT_DIR, "item_index.csv"
)
RESULT_PATH = os.path.join(
    OUTPUT_DIR, "top5_item_similarity.csv"
)


def main():
    print("Memuat matriks Item-User...")

    if not os.path.isfile(MATRIX_PATH):
        raise FileNotFoundError(
            f"Matriks tidak ditemukan: {MATRIX_PATH}"
        )

    if not os.path.isfile(ITEM_INDEX_PATH):
        raise FileNotFoundError(
            f"Indeks Item tidak ditemukan: {ITEM_INDEX_PATH}"
        )

    matrix = load_npz(MATRIX_PATH).tocsr()

    item_index = (
        pd.read_csv(ITEM_INDEX_PATH)
        .sort_values("column_index")
        .reset_index(drop=True)
    )

    if matrix.shape[0] != len(item_index):
        raise ValueError(
            "Jumlah baris matriks tidak sesuai dengan indeks Item."
        )

    item_ids = item_index["itemid"].to_numpy(dtype=np.int64)

    # Derajat Item = jumlah Visitor unik yang berinteraksi dengannya.
    item_degrees = np.asarray(
        matrix.getnnz(axis=1)
    ).ravel()

    # Pemilihan Item acuan reproducible:
    # Item dengan jumlah Visitor unik terbanyak.
    reference_idx = int(np.argmax(item_degrees))
    reference_item_id = int(item_ids[reference_idx])

    print(f"Item acuan       : {reference_item_id}")
    print(f"Jumlah Visitor unik: {item_degrees[reference_idx]:,}")

    # PROYEKSI ITEM-ITEM:
    # P[i,j] menghitung jumlah Visitor bersama antara Item i dan j.
    # Hitung satu baris proyeksi saja agar tidak membangun seluruh
    # matriks Item-Item sekaligus.
    print("Menghitung bobot proyeksi Item-Item...")

    projected_row = (
        matrix[reference_idx] @ matrix.T
    ).toarray().ravel()

    # Cosine similarity dari bobot proyeksi:
    # common_visitors / sqrt(degree_item_i * degree_item_j)
    denominator = np.sqrt(
        item_degrees[reference_idx] * item_degrees
    )

    scores = np.divide(
        projected_row,
        denominator,
        out=np.zeros_like(projected_row, dtype=float),
        where=denominator > 0
    )

    # Item acuan tidak boleh menjadi rekomendasi untuk dirinya sendiri.
    scores[reference_idx] = 0.0

    candidate_indices = np.flatnonzero(scores > 0)

    # Urutkan dari similarity terbesar. Jika nilainya sama,
    # gunakan ID Item untuk menentukan urutan secara konsisten.
    ranked_indices = candidate_indices[
        np.lexsort((
            item_ids[candidate_indices],
            -scores[candidate_indices]
        ))
    ]

    top_indices = ranked_indices[:5]

    if len(top_indices) < 5:
        print(
            f"Peringatan: hanya ditemukan {len(top_indices)} "
            "Item dengan similarity positif."
        )

    results = pd.DataFrame({
        "reference_itemid": reference_item_id,
        "similar_itemid": item_ids[top_indices],
        "similarity": scores[top_indices],
        "similar_item_index": top_indices
    })

    results.to_csv(RESULT_PATH, index=False)

    print("\nTOP 5 ITEM DARI PROYEKSI ITEM-ITEM")
    print("-" * 65)
    print(results.to_string(
        index=False,
        formatters={"similarity": lambda x: f"{x:.8f}"}
    ))
    print("-" * 65)
    print(f"Hasil disimpan ke: {RESULT_PATH}")


if __name__ == "__main__":
    main()