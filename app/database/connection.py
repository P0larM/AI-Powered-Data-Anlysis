import os

import psycopg
from dotenv import load_dotenv

# Reads a .env file in the project root, if there is one.
load_dotenv()


def _connect(variable):
    url = os.environ.get(variable)
    if not url:
        raise RuntimeError(f"Set {variable} in your environment or .env file.")
    # autocommit=True means we control transactions explicitly with
    # `with conn.transaction():` instead of psycopg opening one silently.
    return psycopg.connect(url, autocommit=True)


def get_ingest_connection():
    """Can create schemas/tables and load data. Never give this to AI-generated SQL."""
    return _connect("INGEST_DATABASE_URL")


def get_query_connection():
    """Has no privileges of its own. Used only by run_user_query()."""
    return _connect("QUERY_DATABASE_URL")
