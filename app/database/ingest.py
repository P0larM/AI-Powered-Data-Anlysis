import io

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
)
from psycopg import sql
from psycopg.types.json import Jsonb

from app.data.cleaner import standardize_column_name
from app.data.profiler import profile_dataset
from app.database.tenancy import ensure_user_schema, role_name, schema_name

MAX_IDENTIFIER_LENGTH = 63  # Postgres silently truncates anything longer
COPY_CHUNK_ROWS = 50_000


# ---------- pure helpers (no database needed) --------------------------------

def safe_identifiers(names):
    """Safe, unique, <=63-char snake_case identifiers."""
    result, seen = [], set()
    for name in names:
        base = standardize_column_name(name)[:MAX_IDENTIFIER_LENGTH]
        candidate, counter = base, 2
        while candidate in seen:
            suffix = f"_{counter}"
            candidate = base[:MAX_IDENTIFIER_LENGTH - len(suffix)] + suffix
            counter += 1
        seen.add(candidate)
        result.append(candidate)
    return result


def pg_type_for(series):
    """Map a pandas column to a Postgres type."""
    if is_bool_dtype(series):
        return "BOOLEAN"
    if is_integer_dtype(series):
        return "BIGINT"
    if is_float_dtype(series):
        return "DOUBLE PRECISION"
    if is_datetime64_any_dtype(series):
        return "TIMESTAMP"
    return "TEXT"


def prepare_for_load(df):
    """
    Pandas stores integers with missing values as floats (39.0). Postgres
    rejects "39.0" in a BIGINT column, so turn whole-number float columns
    back into nullable integers.
    """
    df = df.copy()
    for column in df.columns:
        series = df[column]
        if not is_float_dtype(series):
            continue
        series = series.replace([np.inf, -np.inf], np.nan)
        values = series.dropna()
        if len(values) and (values % 1 == 0).all() and values.abs().max() < 9e18:
            series = series.astype("Int64")
        df[column] = series
    return df


# ---------- metadata ---------------------------------------------------------

def _ensure_meta_table(cur):
    cur.execute("CREATE SCHEMA IF NOT EXISTS app_meta")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS app_meta.datasets (
            id                BIGSERIAL PRIMARY KEY,
            user_id           TEXT        NOT NULL,
            table_name        TEXT        NOT NULL,
            original_filename TEXT,
            row_count         BIGINT      NOT NULL,
            columns           JSONB       NOT NULL,
            profile           JSONB       NOT NULL,
            created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (user_id, table_name)
        )
    """)


def get_user_datasets(conn, user_id):
    """
    Everything the app knows about a user's tables: names, row counts,
    column types and the stored profile. This is what you'll put in the
    LLM prompt, and its table names are the allow-list for query validation.
    """
    schema_name(user_id)  # validates user_id
    with conn.transaction():
        with conn.cursor() as cur:
            _ensure_meta_table(cur)
            cur.execute(
                "SELECT table_name, original_filename, row_count, columns, profile "
                "FROM app_meta.datasets WHERE user_id = %s ORDER BY created_at",
                (str(user_id),),
            )
            rows = cur.fetchall()

    return [
        {
            "table_name": r[0],
            "original_filename": r[1],
            "row_count": r[2],
            "columns": r[3],
            "profile": r[4],
        }
        for r in rows
    ]


def delete_dataset(conn, user_id, table_name):
    """
    Remove a dataset from the user's schema and the metadata table.
    """
    schema = schema_name(user_id)
    table = safe_identifiers([table_name])[0]
    qualified = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))

    with conn.transaction():
        with conn.cursor() as cur:
            _ensure_meta_table(cur)
            # Remove from metadata
            cur.execute(
                "DELETE FROM app_meta.datasets WHERE user_id = %s AND table_name = %s",
                (str(user_id), table),
            )
            # Drop the physical table
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(qualified))


# ---------- ingestion --------------------------------------------------------

def ingest_dataframe(conn, df, user_id, table_name,
                     original_filename=None, if_exists="fail"):
    """
    Load a cleaned DataFrame into the user's own schema.

    Parameters:
        conn:      Connection from get_ingest_connection().
        df:        DataFrame already passed through clean_dataframe().
        user_id:   Lowercase letters/digits/underscores.
        table_name: Desired table name (will be sanitized).
        if_exists: "fail" or "replace".

    Everything happens in one transaction: if anything fails, nothing is kept.
    """
    if if_exists not in ("fail", "replace"):
        raise ValueError("if_exists must be 'fail' or 'replace'.")

    schema, role = schema_name(user_id), role_name(user_id)
    table = safe_identifiers([table_name])[0]

    df = df.copy()
    df.columns = safe_identifiers(df.columns)
    df = prepare_for_load(df)

    pg_types = {column: pg_type_for(df[column]) for column in df.columns}
    profile = profile_dataset(df)
    column_info = [
        {"name": c, "pg_type": pg_types[c], "type": profile["columns"][c]["type"]}
        for c in df.columns
    ]

    qualified = sql.SQL("{}.{}").format(sql.Identifier(schema), sql.Identifier(table))

    with conn.transaction():
        with conn.cursor() as cur:
            _ensure_meta_table(cur)
            ensure_user_schema(cur, user_id)

            cur.execute(
                "SELECT 1 FROM app_meta.datasets WHERE user_id = %s AND table_name = %s",
                (str(user_id), table),
            )
            if cur.fetchone() is not None:
                if if_exists == "fail":
                    raise ValueError(f"Table '{table}' already exists for user '{user_id}'.")
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(qualified))
                cur.execute(
                    "DELETE FROM app_meta.datasets WHERE user_id = %s AND table_name = %s",
                    (str(user_id), table),
                )

            definitions = sql.SQL(", ").join(
                sql.SQL("{} {}").format(sql.Identifier(c), sql.SQL(pg_types[c]))
                for c in df.columns
            )
            cur.execute(sql.SQL("CREATE TABLE {} ({})").format(qualified, definitions))

            copy_statement = sql.SQL("COPY {} ({}) FROM STDIN WITH (FORMAT csv, NULL '\\N')").format(
                qualified,
                sql.SQL(", ").join(sql.Identifier(c) for c in df.columns),
            )
            with cur.copy(copy_statement) as copy:
                for start in range(0, len(df), COPY_CHUNK_ROWS):
                    buffer = io.StringIO()
                    df.iloc[start:start + COPY_CHUNK_ROWS].to_csv(
                        buffer, index=False, header=False, na_rep="\\N"
                    )
                    copy.write(buffer.getvalue())

            cur.execute(sql.SQL("GRANT SELECT ON TABLE {} TO {}").format(
                qualified, sql.Identifier(role)))

            cur.execute(
                "INSERT INTO app_meta.datasets "
                "(user_id, table_name, original_filename, row_count, columns, profile) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (str(user_id), table, original_filename, len(df),
                 Jsonb(column_info), Jsonb(profile)),
            )

    return {"schema": schema, "table": table, "rows": int(len(df)), "columns": column_info}
