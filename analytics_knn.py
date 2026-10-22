import os
import numpy as np
import pandas as pd

from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

OUTPUT_DIR = "outputs"
DATA_DIR = os.path.join("data", "processed")

TRAIN_CSV = os.path.join(DATA_DIR, "events_training.csv")
TEST_CSV = os.path.join(DATA_DIR, "events_testing.csv")

MATRIX_PATH = os.path.join(OUTPUT_DIR, "user_item_vsm.npz")
VISITOR_INDEX_PATH = os.path.join(OUTPUT_DIR, "visitor_index.csv")
ITEM_INDEX_PATH = os.path.join(OUTPUT_DIR, "item_index.csv")

RECOMMENDATIONS_PATH = os.path.join(
    OUTPUT_DIR, "knn_recommendations.csv"
)
NEIGHBORS_PATH = os.path.join(
    OUTPUT_DIR, "knn_neighbors.csv"
)
METRICS_PATH = os.path.join(
    OUTPUT_DIR, "knn_metrics.csv"
)
DIAGNOSTICS_PATH = os.path.join(
    OUTPUT_DIR, "knn_diagnostics.csv"
)

RANDOM_SEED = 123
K = 5
TOP_N = 5


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    required_files = [
        TRAIN_CSV,
        TEST_CSV,
        MATRIX_PATH,
        VISITOR_INDEX_PATH,
        ITEM_INDEX_PATH,
    ]

    for path in required_files:
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"File tidak ditemukan: {path}"
            )

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
            "Dimensi matriks tidak cocok dengan file indeks."
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

    print("Membaca interaksi periode testing...")

    test_df = pd.read_csv(
        TEST_CSV,
        usecols=["visitorid", "itemid"],
        dtype={
            "visitorid": "int64",
            "itemid": "int64",
        },
    )

    test_df = test_df[
        test_df["visitorid"].isin(visitor_to_row)
        & test_df["itemid"].isin(item_to_col)
    ]

    relevant_by_user = {}

    for visitor_id, group in test_df.groupby("visitorid"):
        visitor_id = int(visitor_id)
        user_row = visitor_to_row[visitor_id]

        seen_columns = set(
            X.getrow(user_row).indices.tolist()
        )

        test_items = set(
            group["itemid"].astype(int).unique()
        )

        unseen_relevant = {
            item_id
            for item_id in test_items
            if item_to_col[item_id] not in seen_columns
        }

        if unseen_relevant:
            relevant_by_user[visitor_id] = unseen_relevant

    eligible_users = sorted(relevant_by_user.keys())

    if not eligible_users:
        raise ValueError(
            "Tidak ada User dengan Item relevan pada periode testing."
        )

    users_with_history = [
        uid
        for uid in eligible_users
        if X.getrow(visitor_to_row[uid]).nnz >= TOP_N
    ]

    if not users_with_history:
        raise ValueError(
            "Tidak ditemukan User dengan minimal lima Item "
            "di training dan Item relevan di testing."
        )

    rng = np.random.default_rng(RANDOM_SEED)

    candidate_order = rng.permutation(
        np.asarray(users_with_history, dtype=np.int64)
    )

    selected_user_id = None
    selected_row = None
    neighbor_rows = None
    neighbor_scores = None
    item_scores = None
    top_columns = None

    print("Mencari User secara acak dengan seed:", RANDOM_SEED)

    for candidate_user in candidate_order:
        candidate_user = int(candidate_user)
        row = visitor_to_row[candidate_user]
        user_vector = X.getrow(row)

        similarities = cosine_similarity(
            user_vector, X
        ).ravel()

        similarities[row] = 0.0

        candidate_neighbors = np.flatnonzero(
            similarities > 0
        )

        if len(candidate_neighbors) < K:
            continue

        neighbor_order = candidate_neighbors[
            np.lexsort(
                (
                    candidate_neighbors,
                    -similarities[candidate_neighbors],
                )
            )
        ]

        local_neighbor_rows = neighbor_order[:K]
        local_neighbor_scores = similarities[
            local_neighbor_rows
        ]

        neighbor_matrix = X[local_neighbor_rows]

        local_item_scores = np.asarray(
            neighbor_matrix.T @ local_neighbor_scores
        ).ravel()

        local_item_scores[user_vector.indices] = 0.0

        candidate_items = np.flatnonzero(
            local_item_scores > 0
        )

        if len(candidate_items) < TOP_N:
            continue

        local_top_columns = candidate_items[
            np.lexsort(
                (
                    item_ids[candidate_items],
                    -local_item_scores[candidate_items],
                )
            )
        ][:TOP_N]

        selected_user_id = candidate_user
        selected_row = row
        neighbor_rows = local_neighbor_rows
        neighbor_scores = local_neighbor_scores
        item_scores = local_item_scores
        top_columns = local_top_columns

        break

    if selected_user_id is None:
        raise ValueError(
            "Tidak ditemukan User yang memenuhi kriteria "
            "rekomendasi. Periksa data training dan testing."
        )

    relevant_items = relevant_by_user[selected_user_id]

    print("\nUSER TERPILIH")
    print("-" * 60)
    print("Random seed                  :", RANDOM_SEED)
    print("Visitor ID                   :", selected_user_id)
    print(
        "Item unik di training        :",
        X.getrow(selected_row).nnz,
    )
    print(
        "Item relevan baru di testing :",
        len(relevant_items),
    )

    neighbors_df = pd.DataFrame({
        "neighbor_rank": np.arange(1, K + 1),
        "neighbor_visitorid": visitor_ids[neighbor_rows],
        "cosine_similarity": neighbor_scores,
    })

    neighbors_df.to_csv(
        NEIGHBORS_PATH, index=False
    )

    print("\nK-NEAREST NEIGHBORS")
    print(
        neighbors_df.to_string(
            index=False,
            formatters={
                "cosine_similarity": lambda x: f"{x:.6f}"
            },
        )
    )

    recommended_item_ids = item_ids[top_columns]

    recommendation_relevance = np.array([
        int(int(item_id) in relevant_items)
        for item_id in recommended_item_ids
    ], dtype=int)

    recommendations_df = pd.DataFrame({
        "visitorid": selected_user_id,
        "rank": np.arange(1, TOP_N + 1),
        "itemid": recommended_item_ids,
        "score": item_scores[top_columns],
        "relevant_in_testing": recommendation_relevance,
    })

    recommendations_df.to_csv(
        RECOMMENDATIONS_PATH, index=False
    )

    print("\nTOP 5 REKOMENDASI ITEM")
    print(
        recommendations_df.to_string(
            index=False,
            formatters={"score": lambda x: f"{x:.6f}"},
        )
    )

    print("\nTIGA ITEM UNTUK JAWABAN SOAL 2.2.1(b)")
    for row in recommendations_df.head(3).itertuples(
        index=False
    ):
        print(
            f"Rank {row.rank}: Item {row.itemid} "
            f"(skor={row.score:.6f})"
        )

    recommended_set = set(
        map(int, recommended_item_ids)
    )

    hits = sorted(
        recommended_set.intersection(relevant_items)
    )

    print("\nDIAGNOSTIK EVALUASI")
    print("-" * 60)
    print(
        "Jumlah Item relevan di testing:",
        len(relevant_items),
    )

    relevant_items_sorted = sorted(relevant_items)

    print(
        "Item relevan di testing (maks. 30):",
        relevant_items_sorted[:30],
    )

    print(
        "Item rekomendasi:",
        recommended_item_ids.astype(int).tolist(),
    )

    print("Item yang cocok (hits):", hits)
    print("Jumlah hits:", len(hits))

    hit_at_5 = float(
        np.any(recommendation_relevance == 1)
    )

    discounts = 1.0 / np.log2(
        np.arange(2, TOP_N + 2)
    )

    dcg_at_5 = float(
        np.sum(recommendation_relevance * discounts)
    )

    ideal_length = min(
        len(relevant_items), TOP_N
    )

    ideal_discounts = 1.0 / np.log2(
        np.arange(2, ideal_length + 2)
    )

    idcg_at_5 = float(
        np.sum(ideal_discounts)
    )

    ndcg_at_5 = (
        dcg_at_5 / idcg_at_5
        if idcg_at_5 > 0
        else 0.0
    )

    metrics_df = pd.DataFrame([{
        "random_seed": RANDOM_SEED,
        "visitorid": selected_user_id,
        "k_neighbors": K,
        "recommendation_cutoff": TOP_N,
        "relevant_test_items": len(relevant_items),
        "hits_in_top5": int(recommendation_relevance.sum()),
        "hit_at_5": hit_at_5,
        "dcg_at_5": dcg_at_5,
        "idcg_at_5": idcg_at_5,
        "ndcg_at_5": ndcg_at_5,
    }])

    metrics_df.to_csv(
        METRICS_PATH, index=False
    )

    diagnostic_rows = []

    for item_id in relevant_items_sorted:
        diagnostic_rows.append({
            "visitorid": selected_user_id,
            "itemid": item_id,
            "source": "testing_relevant",
            "recommended_rank": np.nan,
            "relevant": 1,
        })

    for rank, item_id in enumerate(
        recommended_item_ids, start=1
    ):
        diagnostic_rows.append({
            "visitorid": selected_user_id,
            "itemid": int(item_id),
            "source": "recommendation",
            "recommended_rank": rank,
            "relevant": int(int(item_id) in relevant_items),
        })

    pd.DataFrame(diagnostic_rows).to_csv(
        DIAGNOSTICS_PATH, index=False
    )

    print("\nEVALUASI REKOMENDASI")
    print("-" * 60)
    print(f"Hit@5  : {hit_at_5:.4f}")
    print(f"DCG@5  : {dcg_at_5:.6f}")
    print(f"IDCG@5 : {idcg_at_5:.6f}")
    print(f"NDCG@5 : {ndcg_at_5:.4f}")

    print("\nFile hasil:")
    print("-", NEIGHBORS_PATH)
    print("-", RECOMMENDATIONS_PATH)
    print("-", METRICS_PATH)
    print("-", DIAGNOSTICS_PATH)


if __name__ == "__main__":
    main()
