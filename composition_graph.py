from getpass import getpass
from neo4j import GraphDatabase

URI = "neo4j://127.0.0.1:7687"
DATABASE = "neo4j"
USERNAME = "neo4j"

QUERY = """
MATCH (n:Visitor)
RETURN 'Visitor' AS component, count(n) AS total

UNION ALL

MATCH (n:Item)
RETURN 'Item' AS component, count(n) AS total

UNION ALL

MATCH ()-[r:VIEWED]->()
RETURN 'VIEWED' AS component, count(r) AS total

UNION ALL

MATCH ()-[r:ADDED_TO_CART]->()
RETURN 'ADDED_TO_CART' AS component, count(r) AS total

UNION ALL

MATCH ()-[r:BOUGHT]->()
RETURN 'BOUGHT' AS component, count(r) AS total

ORDER BY component
"""


def main():
    password = getpass("Password Neo4j: ")

    with GraphDatabase.driver(
        URI, auth=(USERNAME, password)
    ) as driver:
        driver.verify_connectivity()

        records, summary, keys = driver.execute_query(
            QUERY,
            database_=DATABASE
        )

        results = {
            record["component"]: record["total"]
            for record in records
        }

        print("\nKOMPOSISI GRAF RETAILROCKET - DATA TRAINING")
        print("-" * 48)
        print(f"{'Komponen':<20}{'Jumlah':>15}")
        print("-" * 48)

        for component in sorted(results):
            print(
                f"{component:<20}"
                f"{results[component]:>15,}"
            )

        total_nodes = (
            results["Visitor"] + results["Item"]
        )

        total_relationships = sum(
            results[name]
            for name in [
                "VIEWED",
                "ADDED_TO_CART",
                "BOUGHT"
            ]
        )

        print("-" * 48)
        print(f"{'TOTAL NODE':<20}{total_nodes:>15,}")
        print(
            f"{'TOTAL RELATIONSHIP':<20}"
            f"{total_relationships:>15,}"
        )


if __name__ == "__main__":
    main()
