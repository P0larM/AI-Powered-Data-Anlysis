"""End-to-end smoke test: load Adult into Postgres, then run good and bad queries.

Run from the project root:  python -m app.db_demo
"""
from app.data.cleaner import clean_dataframe
from app.data.loader import load_dataset
from app.database.connection import get_ingest_connection, get_query_connection
from app.database.ingest import get_user_datasets, ingest_dataframe
from app.database.queries import QueryRejected, run_user_query
from app.main import ADULT_COLUMNS, DATASET_PATH

USER_ID = "demo"

QUERIES = [
    ("good", "SELECT sex, COUNT(*) AS n, ROUND(AVG(age), 1) AS avg_age FROM adult GROUP BY sex"),
    ("good", "SELECT COUNT(*) AS male_count FROM adult WHERE sex = 'Male'"),
    ("good", "SELECT COUNT(*) AS missing_workclass FROM adult WHERE workclass IS NULL"),
    ("good", "WITH rich AS (SELECT * FROM adult WHERE income = '>50K') SELECT COUNT(*) FROM rich"),
    ("bad ", "DROP TABLE adult"),
    ("bad ", "SELECT * FROM adult; DELETE FROM adult"),
    ("bad ", "SELECT * FROM user_other.adult"),
    ("bad ", "SELECT * FROM app_meta.datasets"),
    ("bad ", "SELECT pg_sleep(60)"),
    ("bad ", "SELECT * INTO stolen FROM adult"),
]


def main():
    df = clean_dataframe(load_dataset(DATASET_PATH, column_names=ADULT_COLUMNS))

    with get_ingest_connection() as ingest_conn:
        info = ingest_dataframe(
            ingest_conn, df, USER_ID, "adult",
            original_filename="adult.data", if_exists="replace",
        )
        print(f"Loaded {info['rows']} rows into {info['schema']}.{info['table']}")
        tables = [d["table_name"] for d in get_user_datasets(ingest_conn, USER_ID)]

    with get_query_connection() as query_conn:
        for label, query in QUERIES:
            print(f"\n[{label}] {query}")
            try:
                result = run_user_query(query_conn, USER_ID, query, tables)
                print("   ->", result["columns"], result["rows"][:5])
            except QueryRejected as error:
                print("   -> REJECTED:", error)
            except Exception as error:  # database-level errors
                print("   -> DATABASE ERROR:", type(error).__name__, error)


if __name__ == "__main__":
    main()
