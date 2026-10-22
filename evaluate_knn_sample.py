
import os
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "outputs"
DATA_DIR = os.path.join("data", "processed")

TEST_CSV = os.path.join(DATA_DIR, "events_testing.csv")
MATRIX_PATH = os.path.join(OUTPUT_DIR, "user_item_vsm.npz")
VISITOR_INDEX_PATH = os.path.join(OUTPUT_DIR, "visitor_index.csv")
ITEM_INDEX_PATH = os.path.join(OUTPUT_DIR, "item_index.csv")

USER_RESULTS_PATH = os.path.join(
    OUTPUT_DIR, "knn_batch_user_results.csv"
)
REC_RESULTS_PATH = os.path.join(
    OUTPUT_DIR, "knn_batch_recommendations.csv"
)
SUMMARY_PATH = os.path.join(
    OUTPUT_DIR, "knn_batch_summary.csv"
)

RANDOM_SEED = 2026
SAMPLE_SIZE = 100
K = 5
TOP_N = 5


def calculate_ndcg(relevance, total_relevant):
    """Menghitung NDCG@5 dengan relevansi biner."""
    relevance = np.asarray(relevance, dtype=float)

    discounts = 1.0 / np.log2(
        np.arange(2, len(relevance) + 2)
    )

    dcg = float(np.sum(relevance * discounts))

    ideal_length = min(total_relevant, len(relevance))

    if ideal_length == 0:
        return 0.0, dcg, 0.0

    ideal_discounts = 1.0 / np.log2(
        np.arange(2, ideal_length + 2)
    )

    idcg = float(np.sum(ideal_discounts))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return ndcg, dcg, idcg


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------
    # 1. Validate files and load training matrix
    # --------------------------------------------------------
    required_files = [
        TEST_CSV,
        MATRIX_PATH,
        VISITOR_INDEX_PATH,
        ITEM_INDEX_PATH,
    ]

    for path in required_files:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")

    print("Memuat matriks User-Item...")
    X = load_npz(MATRIX_PATH).tocsr()

    visitor_index = (
        pd.read_csv(VISITOR_INDEX_PATH)
        .sort_values("row_index")
        .reset_index(drop=True)
    )

    item_index = (
        pd.read_csv(ITEM_INDEX_PATH)
        .sort_values("column_index")
        .reset_index(drop=True)
    )

    visitor_ids = visitor_index["visitorid"].to_numpy(
        dtype=np.int64
    )
    item_ids = item_index["itemid"].to_numpy(
        dtype=np.int64
    )

    if X.shape != (len(visitor_ids), len(item_ids)):
        raise ValueError(
            "Shape matriks tidak cocok dengan file indeks."
        )

    visitor_to_row = {
        int(visitor_id): row
        for row, visitor_id in enumerate(visitor_ids)
    }

    item_to_col = {
        int(item_id): col
        for col, item_id in enumerate(item_ids)
    }

    print("Shape matriks training:", X.shape)

    # --------------------------------------------------------
    # 2. Create relevance sets from the testing period
    # --------------------------------------------------------
    print("Membaca periode testing...")

    test_df = pd.read_csv(
        TEST_CSV,
        usecols=["visitorid", "itemid"],
        dtype={
            "visitorid": "int64",
            "itemid": "int64",
        },
    )

    # Evaluasi warm-start: hanya Visitor dan Item yang dikenal
    # oleh matriks training yang menjadi kandidat evaluasi.
    test_df = test_df[
        test_df["visitorid"].isin(visitor_to_row)
        & test_df["itemid"].isin(item_to_col)
    ]

    relevant_by_user = {}

    for visitor_id, group in test_df.groupby("visitorid"):
        visitor_id = int(visitor_id)
        row = visitor_to_row[visitor_id]

        seen_items = set(X.getrow(row).indices.tolist())
        test_items = {
            item_to_col[int(item_id)]
            for item_id in group["itemid"].unique()
        }

        relevant_columns = test_items - seen_items

        if relevant_columns:
            relevant_by_user[visitor_id] = relevant_columns

    # Kriteria eligibility ditentukan sebelum rekomendasi dihitung:
    # minimal 5 Item unik pada training dan minimal 1 Item relevan
    # yang belum terlihat pada testing.
    eligible_users = sorted(
        uid
        for uid, relevant in relevant_by_user.items()
        if X.getrow(visitor_to_row[uid]).nnz >= K
        and len(relevant) > 0
    )

    if not eligible_users:
        raise ValueError(
            "Tidak ada Visitor yang memenuhi kriteria evaluasi."
        )

    # --------------------------------------------------------
    # 3. Reproducibly sample users; never filter by their scores
    # --------------------------------------------------------
    rng = np.random.default_rng(RANDOM_SEED)

    sample_size = min(SAMPLE_SIZE, len(eligible_users))

    sampled_users = rng.choice(
        np.asarray(eligible_users, dtype=np.int64),
        size=sample_size,
        replace=False,
    )

    print("\nPENGATURAN EVALUASI")
    print("-" * 55)
    print("Random seed:", RANDOM_SEED)
    print("Ukuran sampel:", sample_size)
    print("Eligible Visitor:", len(eligible_users))
    print("K tetangga:", K)
    print("Jumlah rekomendasi:", TOP_N)

    # --------------------------------------------------------
    # 4. Evaluate each sampled user
    # --------------------------------------------------------
    user_results = []
    all_recommendations = []

    for sample_position, user_id in enumerate(
        sampled_users, start=1
    ):
        user_id = int(user_id)
        user_row = visitor_to_row[user_id]
        user_vector = X.getrow(user_row)

        relevant_columns = relevant_by_user[user_id]
        relevant_item_ids = {
            int(item_ids[col]) for col in relevant_columns
        }

        # Cosine similarity to every training user.
        similarities = cosine_similarity(
            user_vector, X
        ).ravel()

        # Exclude the target user.
        similarities[user_row] = -1.0

        # Always select the top K other users. Zero-similarity
        # neighbors are allowed when fewer than K users overlap.
        candidates = np.arange(X.shape[0])
        candidates = candidates[candidates != user_row]

        neighbor_order = candidates[
            np.lexsort(
                (
                    candidates,
                    -similarities[candidates],
                )
            )
        ]

        neighbor_rows = neighbor_order[:K]
        neighbor_scores = similarities[neighbor_rows]
        neighbor_scores = np.maximum(neighbor_scores, 0.0)

        # Weighted item scores from the K neighbors.
        item_scores = np.asarray(
            X[neighbor_rows].T @ neighbor_scores
        ).ravel()

        # Do not recommend items already seen by the target user.
        item_scores[user_vector.indices] = 0.0

        candidate_items = np.flatnonzero(item_scores > 0)

        ranked_items = candidate_items[
            np.lexsort(
                (
                    item_ids[candidate_items],
                    -item_scores[candidate_items],
                )
            )
        ]

        top_columns = ranked_items[:TOP_N]
        recommended_item_ids = item_ids[top_columns]

        # Pad relevance with zeros if fewer than five items
        # can be recommended. Missing ranks earn no gain.
        relevance = np.zeros(TOP_N, dtype=int)

        for rank, item_id in enumerate(recommended_item_ids):
            relevance[rank] = int(
                int(item_id) in relevant_item_ids
            )

        hit_at_5 = float(np.any(relevance == 1))

        ndcg_at_5, dcg_at_5, idcg_at_5 = calculate_ndcg(
            relevance,
            total_relevant=len(relevant_item_ids),
        )

        hits = sorted(
            set(map(int, recommended_item_ids))
            & relevant_item_ids
        )

        user_results.append({
            "sample_position": sample_position,
            "random_seed": RANDOM_SEED,
            "visitorid": user_id,
            "training_unique_items": int(user_vector.nnz),
            "relevant_testing_items": len(relevant_item_ids),
            "neighbors_found": len(neighbor_rows),
            "recommendations_generated": len(recommended_item_ids),
            "hits_in_top5": len(hits),
            "hit_at_5": hit_at_5,
            "dcg_at_5": dcg_at_5,
            "idcg_at_5": idcg_at_5,
            "ndcg_at_5": ndcg_at_5,
        })

        for rank, item_id in enumerate(
            recommended_item_ids, start=1
        ):
            all_recommendations.append({
                "visitorid": user_id,
                "rank": rank,
                "itemid": int(item_id),
                "score": float(item_scores[item_to_col[int(item_id)]]),
                "relevant_in_testing": int(
                    int(item_id) in relevant_item_ids
                ),
            })

        if sample_position % 10 == 0:
            print(
                f"Memproses Visitor: "
                f"{sample_position}/{sample_size}"
            )

    # --------------------------------------------------------
    # 5. Save detailed outputs
    # --------------------------------------------------------
    results_df = pd.DataFrame(user_results)
    recommendations_df = pd.DataFrame(all_recommendations)

    results_df.to_csv(USER_RESULTS_PATH, index=False)
    recommendations_df.to_csv(REC_RESULTS_PATH, index=False)

    # --------------------------------------------------------
    # 6. Overall metrics across sampled users
    # --------------------------------------------------------
    summary = pd.DataFrame([{
        "random_seed": RANDOM_SEED,
        "sample_size": len(results_df),
        "eligible_users": len(eligible_users),
        "k_neighbors": K,
        "recommendation_cutoff": TOP_N,
        "users_with_hit_at_5": int(
            (results_df["hit_at_5"] > 0).sum()
        ),
        "hit_at_5_rate": float(
            results_df["hit_at_5"].mean()
        ),
        "mean_ndcg_at_5": float(
            results_df["ndcg_at_5"].mean()
        ),
        "mean_dcg_at_5": float(
            results_df["dcg_at_5"].mean()
        ),
        "mean_relevant_test_items": float(
            results_df["relevant_testing_items"].mean()
        ),
        "mean_recommendations_generated": float(
            results_df["recommendations_generated"].mean()
        ),
    }])

    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nRINGKASAN EVALUASI SAMPEL")
    print("-" * 60)
    print(summary.T.to_string(header=False))
    print("-" * 60)

    print("\nDistribusi Hit@5:")
    print(results_df["hit_at_5"].value_counts().sort_index())

    print("\nSepuluh hasil pertama:")
    print(
        results_df[
            [
                "visitorid",
                "relevant_testing_items",
                "recommendations_generated",
                "hits_in_top5",
                "hit_at_5",
                "ndcg_at_5",
            ]
        ].head(10).to_string(index=False)
    )

    print("\nFile hasil:")
    print("-", USER_RESULTS_PATH)
    print("-", REC_RESULTS_PATH)
    print("-", SUMMARY_PATH)


if __name__ == "__main__":
    main()
