import argparse
import math
import os
import sys
from getpass import getpass

import pandas as pd
from neo4j import GraphDatabase
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Impor dataset RetailRocket (events.csv) ke Neo4j."
    )

    parser.add_argument(
        "-u", "--user",
        default="neo4j",
        help="Username Neo4j (default: neo4j)"
    )
    parser.add_argument(
        "-p", "--password",
        default=None,
        help="Password Neo4j; jika kosong, akan diminta secara interaktif"
    )
    parser.add_argument(
        "-b", "--bolt",
        default="neo4j://127.0.0.1:7687",
        help="Neo4j URI (default: neo4j://127.0.0.1:7687)"
    )
    parser.add_argument(
        "-s", "--source",
        default="data/raw",
        help="Folder yang berisi events.csv (default: data/raw)"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path CSV input; jika kosong, gunakan events.csv pada source"
    )
    parser.add_argument(
        "--database",
        default="neo4j",
        help="Nama database Neo4j (default: neo4j)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Jumlah baris per batch (default: 10000)"
    )

    return parser.parse_args()


def create_constraints(session):
    session.run(
        "CREATE CONSTRAINT visitor_id_unique IF NOT EXISTS "
        "FOR (v:Visitor) REQUIRE v.visitorId IS UNIQUE"
    ).consume()

    session.run(
        "CREATE CONSTRAINT item_id_unique IF NOT EXISTS "
        "FOR (i:Item) REQUIRE i.itemId IS UNIQUE"
    ).consume()


def import_batch(tx, batch):
    cypher_query = """
    UNWIND $batch AS row

    MERGE (v:Visitor {visitorId: row.visitorid})
    MERGE (i:Item {itemId: row.itemid})

    WITH v, i, row

    CALL {
        WITH v, i, row
        WITH v, i, row WHERE row.event = 'view'
        CREATE (v)-[:VIEWED {timestamp: row.timestamp}]->(i)

        UNION ALL

        WITH v, i, row
        WITH v, i, row WHERE row.event = 'addtocart'
        CREATE (v)-[:ADDED_TO_CART {timestamp: row.timestamp}]->(i)

        UNION ALL

        WITH v, i, row
        WITH v, i, row WHERE row.event = 'transaction'
        CREATE (v)-[:BOUGHT {
            timestamp: row.timestamp,
            transactionId: row.transactionid
        }]->(i)
    }
    """

    tx.run(cypher_query, batch=batch).consume()


def count_csv_rows(csv_file_path):
    with open(csv_file_path, "r", encoding="utf-8", newline="") as file:
        return max(sum(1 for _ in file) - 1, 0)


def main():
    args = parse_arguments()

    if args.batch_size < 1:
        print("Error: batch size harus lebih besar dari 0.")
        sys.exit(1)

    csv_file_path = (
        args.input
        if args.input
        else os.path.join(args.source, "events.csv")
    )

    if not os.path.isfile(csv_file_path):
        print(f"Error: File '{csv_file_path}' tidak ditemukan!")
        sys.exit(1)

    password = args.password
    if password is None:
        password = getpass("Password Neo4j: ")

    driver = None

    try:
        driver = GraphDatabase.driver(
            args.bolt,
            auth=(args.user, password)
        )

        driver.verify_connectivity()
        print("Koneksi ke Neo4j berhasil.")

        total_rows = count_csv_rows(csv_file_path)
        total_chunks = math.ceil(total_rows / args.batch_size)

        print(f"Membaca file: {csv_file_path}")
        print(f"Total baris data: {total_rows:,}")
        print(f"Ukuran batch: {args.batch_size:,}")

        with driver.session(database=args.database) as session:
            create_constraints(session)

            chunks = pd.read_csv(
                csv_file_path,
                chunksize=args.batch_size,
                dtype={"transactionid": "string"}
            )

            print("Memulai proses impor data ke Neo4j...")

            for chunk in tqdm(
                chunks,
                total=total_chunks,
                desc="Mengimpor Batch Data"
            ):
                chunk["transactionid"] = (
                    chunk["transactionid"].fillna("").astype(str)
                )

                records = chunk.to_dict(orient="records")
                session.execute_write(import_batch, records)

        print("Proses impor berhasil.")

    except Exception as error:
        print(f"Gagal melakukan impor: {error}")
        sys.exit(1)

    finally:
        if driver is not None:
            driver.close()


if __name__ == "__main__":
    main()
