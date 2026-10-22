import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, save_npz
from sklearn.metrics.pairwise import cosine_similarity


SOURCE = os.path.join(
    "data", "processed", "events_training.csv"
)
OUTPUT_DIR = "outputs"

MATRIX_PATH = os.path.join(
    OUTPUT_DIR, "user_item_vsm.npz"
)
USER_INDEX_PATH = os.path.join(
    OUTPUT_DIR, "visitor_index.csv"
)
ITEM_INDEX_PATH = os.path.join(
    OUTPUT_DIR, "item_index.csv"
)
PLOT_PATH = os.path.join(
    OUTPUT_DIR, "user_projection.png"
)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(SOURCE):
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {SOURCE}"
        )

    print("Membaca data training...")
    df = pd.read_csv(
        SOURCE,
        usecols=["visitorid", "itemid"]
    )

    # Membuat indeks unik untuk User dan Item.
    user_ids, user_rows = np.unique(
        df["visitorid"].to_numpy(),
        return_inverse=True
    )

    item_ids, item_cols = np.unique(
        df["itemid"].to_numpy(),
        return_inverse=True
    )

    # Membuat sparse User-Item matrix.
    X = coo_matrix(
        (
            np.ones(len(df), dtype=np.float32),
            (user_rows, item_cols)
        ),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32
    ).tocsr()

    # Mengubah interaksi berulang menjadi nilai biner 1.
    X.sum_duplicates()
    X.data[:] = 1.0
    X.eliminate_zeros()

    # Menyimpan matriks dan pemetaan ID.
    save_npz(MATRIX_PATH, X, compressed=True)

    pd.DataFrame({
        "row_index": np.arange(len(user_ids)),
        "visitorid": user_ids
    }).to_csv(USER_INDEX_PATH, index=False)

    pd.DataFrame({
        "column_index": np.arange(len(item_ids)),
        "itemid": item_ids
    }).to_csv(ITEM_INDEX_PATH, index=False)

    total_possible = X.shape[0] * X.shape[1]
    density = X.nnz / total_possible

    print("\nKARAKTERISTIK VSM USER")
    print("-" * 45)
    print(f"Jumlah User       : {X.shape[0]:,}")
    print(f"Jumlah Item       : {X.shape[1]:,}")
    print(f"Shape matriks     : {X.shape}")
    print(f"Interaksi unik    : {X.nnz:,}")
    print(f"Jumlah elemen nol : {total_possible - X.nnz:,}")
    print(f"Densitas matriks  : {density:.8%}")
    print(f"Format matriks    : {X.getformat()}")
    print(f"Matriks disimpan  : {MATRIX_PATH}")

    # Visualisasi proyeksi User-User dari 100 User
    # dengan jumlah interaksi unik terbanyak.
    # Ini hanya sampel untuk visualisasi, bukan seluruh graf.
    sample_size = min(100, X.shape[0])
    activity = np.asarray(X.getnnz(axis=1)).ravel()
    sample_rows = np.argsort(activity)[-sample_size:]

    X_sample = X[sample_rows]
    similarities = cosine_similarity(X_sample)

    G = nx.Graph()

    for idx in sample_rows:
        G.add_node(int(user_ids[idx]))

    k = min(3, max(0, sample_size - 1))

    for i in range(sample_size):
        # Mengambil tetangga dengan cosine similarity tertinggi.
        candidates = np.argsort(similarities[i])[::-1]

        added = 0
        for j in candidates:
            if i == j:
                continue

            score = float(similarities[i, j])

            if score <= 0:
                break

            G.add_edge(
                int(user_ids[sample_rows[i]]),
                int(user_ids[sample_rows[j]]),
                weight=score
            )

            added += 1
            if added >= k:
                break

    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(G, seed=42, weight="weight")

    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=False,
        node_size=35,
        width=0.7,
        alpha=0.8
    )

    plt.title(
        "User-User Projection (Top 100 Active Users)"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print(f"\nNode sampel pada graf : {G.number_of_nodes()}")
    print(f"Edge proyeksi sampel  : {G.number_of_edges()}")
    print(f"Plot disimpan        : {PLOT_PATH}")


if __name__ == "__main__":
    main()