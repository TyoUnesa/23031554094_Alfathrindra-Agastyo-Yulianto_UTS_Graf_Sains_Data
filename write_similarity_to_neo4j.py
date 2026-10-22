import os
from getpass import getpass

import pandas as pd
from neo4j import GraphDatabase


URI = "neo4j://127.0.0.1:7687"
DATABASE = "neo4j"
USERNAME = "neo4j"

INPUT_PATH = os.path.join(
    "outputs", "top5_item_similarity.csv"
)

QUERY = """
UNWIND $rows AS row

MATCH (source:Item {itemId: row.reference_itemid})
MATCH (target:Item {itemId: row.similar_itemid})

MERGE (source)-[r:SIMILAR_TO]->(target)
SET r.similarity = row.similarity

RETURN count(r) AS relationships_written
"""


def main():
    if not os.path.isfile(INPUT_PATH):
        raise FileNotFoundError(
            f"File hasil similarity tidak ditemukan: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    required_columns = {
        "reference_itemid",
        "similar_itemid",
        "similarity"
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Kolom CSV harus mencakup: {required_columns}"
        )

    rows = [
        {
            "reference_itemid": int(row.reference_itemid),
            "similar_itemid": int(row.similar_itemid),
            "similarity": float(row.similarity)
        }
        for row in df.itertuples(index=False)
    ]

    password = getpass("Password Neo4j: ")

    with GraphDatabase.driver(
        URI, auth=(USERNAME, password)
    ) as driver:
        driver.verify_connectivity()

        records, _, _ = driver.execute_query(
            QUERY,
            rows=rows,
            database_=DATABASE
        )

        written = records[0]["relationships_written"]

        print(f"Jumlah baris kandidat: {len(rows)}")
        print(f"Relationship diproses: {written}")

        if written != len(rows):
            print(
                "PERINGATAN: Tidak semua pasangan berhasil "
                "diproses. Periksa apakah seluruh Item tersedia "
                "di database training."
            )
        else:
            print(
                "Semua relationship SIMILAR_TO berhasil disimpan."
            )


if __name__ == "__main__":
    main()
