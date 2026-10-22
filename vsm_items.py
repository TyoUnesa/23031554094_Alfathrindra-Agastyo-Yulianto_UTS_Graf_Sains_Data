import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz
from sklearn.metrics.pairwise import cosine_similarity


OUTPUT_DIR = "outputs"

USER_ITEM_PATH = os.path.join(
    OUTPUT_DIR, "user_item_vsm.npz"
)
ITEM_INDEX_PATH = os.path.join(
    OUTPUT_DIR, "item_index.csv"
)
ITEM_USER_PATH = os.path.join(
    OUTPUT_DIR, "item_user_vsm.npz"
)
PLOT_PATH = os.path.join(
    OUTPUT_DIR, "item_projection.png"
)


def main():
    if not os.path.isfile(USER_ITEM_PATH):
        raise FileNotFoundError(
            f"Matriks User-Item tidak ditemukan: {USER_ITEM_PATH}"
        )

    if not os.path.isfile(ITEM_INDEX_PATH):
        raise FileNotFoundError(
            f"Indeks Item tidak ditemukan: {ITEM_INDEX_PATH}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Memuat matriks User-Item...")
    X_user_item = load_npz(USER_ITEM_PATH)

    # Transpose: setiap baris menjadi Item,
    # setiap kolom menjadi User.
    X_item_user = X_user_item.transpose().tocsr()

    # Simpan matriks sparse Item-User.
    save_npz(ITEM_USER_PATH, X_item_user, compressed=True)

    item_ids = pd.read_csv(
        ITEM_INDEX_PATH
    )["itemid"].to_numpy()

    total_possible = (
        X_item_user.shape[0] * X_item_user.shape[1]
    )
    density = X_item_user.nnz / total_possible

    print("\nKARAKTERISTIK VSM ITEM")
    print("-" * 45)
    print(f"Jumlah Item       : {X_item_user.shape[0]:,}")
    print(f"Jumlah User       : {X_item_user.shape[1]:,}")
    print(f"Shape matriks     : {X_item_user.shape}")
    print(f"Interaksi unik    : {X_item_user.nnz:,}")
    print(f"Jumlah elemen nol : {total_possible - X_item_user.nnz:,}")
    print(f"Densitas matriks  : {density:.8%}")
    print(f"Format matriks    : {X_item_user.getformat()}")
    print(f"Matriks disimpan  : {ITEM_USER_PATH}")

    # Proyeksi visual menggunakan maksimal 100 Item
    # dengan jumlah User unik terbanyak.
    sample_size = min(100, X_item_user.shape[0])
    activity = np.asarray(
        X_item_user.getnnz(axis=1)
    ).ravel()

    sample_rows = np.argsort(activity)[-sample_size:]
    X_sample = X_item_user[sample_rows]

    # Hitung kemiripan antar-Item pada sampel.
    similarities = cosine_similarity(X_sample)

    G = nx.Graph()

    for idx in sample_rows:
        G.add_node(int(item_ids[idx]))

    k = min(3, max(0, sample_size - 1))

    for i in range(sample_size):
        candidates = np.argsort(similarities[i])[::-1]
        added = 0

        for j in candidates:
            if i == j:
                continue

            score = float(similarities[i, j])

            if score <= 0:
                break

            G.add_edge(
                int(item_ids[sample_rows[i]]),
                int(item_ids[sample_rows[j]]),
                weight=score
            )

            added += 1
            if added >= k:
                break

    plt.figure(figsize=(12, 9))

    pos = nx.spring_layout(
        G, seed=42, weight="weight"
    )

    nx.draw_networkx(
        G,
        pos=pos,
        with_labels=False,
        node_size=35,
        width=0.7,
        alpha=0.8
    )

    plt.title(
        "Item-Item Projection (Top 100 Active Items)"
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
