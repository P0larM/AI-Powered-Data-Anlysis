import os
import re

from psycopg import sql

_USER_ID = re.compile(r"[a-z0-9_]{1,40}")


def schema_name(user_id):
    """Every user's tables live in their own schema."""
    user_id = str(user_id)
    if not _USER_ID.fullmatch(user_id):
        raise ValueError("user_id may only contain lowercase letters, digits and underscores (max 40).")
    return f"user_{user_id}"


def role_name(user_id):
    """Read-only role that can see only this user's schema."""
    return f"{schema_name(user_id)}_ro"


def ensure_user_schema(cur, user_id):
    """Create the user's schema and read-only role (safe to call repeatedly)."""
    schema, role = schema_name(user_id), role_name(user_id)
    login_role = os.environ.get("QUERY_DB_ROLE", "app_query")

    cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema)))

    cur.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role,))
    if cur.fetchone() is None:
        cur.execute(sql.SQL("CREATE ROLE {} NOLOGIN").format(sql.Identifier(role)))

    cur.execute(sql.SQL("GRANT {} TO {}").format(sql.Identifier(role), sql.Identifier(login_role)))
    cur.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
        sql.Identifier(schema), sql.Identifier(role)))
    cur.execute(sql.SQL("ALTER DEFAULT PRIVILEGES IN SCHEMA {} GRANT SELECT ON TABLES TO {}").format(
        sql.Identifier(schema), sql.Identifier(role)))
