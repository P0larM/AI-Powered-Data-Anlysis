import datetime
import decimal

import sqlglot
from psycopg import sql
from sqlglot import exp
from sqlglot.errors import SqlglotError

from app.database.tenancy import role_name, schema_name


class QueryRejected(ValueError):
    """The SQL was refused before reaching the database."""


# Newer sqlglot versions call this SetOperation, older ones Union.
_SET_OPERATION = getattr(exp, "SetOperation", exp.Union)
_ALLOWED_TOP_LEVEL = (exp.Select, _SET_OPERATION, exp.Subquery)

_FORBIDDEN_NODE_NAMES = [
    "Insert", "Update", "Delete", "Merge", "Drop", "Create", "Alter", "AlterTable",
    "TruncateTable", "Command", "Into", "Lock", "Copy", "Set", "Pragma", "Use",
    "Transaction", "Commit", "Rollback",
]
_FORBIDDEN_NODES = tuple(getattr(exp, n) for n in _FORBIDDEN_NODE_NAMES if hasattr(exp, n))

_BLOCKED_FUNCTIONS = {"set_config", "current_setting", "dblink", "dblink_exec"}
_BLOCKED_FUNCTION_PREFIXES = ("pg_", "lo_")


def _function_name(node):
    if isinstance(node, exp.Anonymous):
        return str(node.name).lower()
    return node.sql_name().lower()


def validate_select(query, allowed_tables):
    """
    Raise QueryRejected unless `query` is a single read-only SELECT that only
    touches `allowed_tables` (unqualified names). Returns the cleaned query.

    This is the first layer. The database role (see tenancy.py) is the layer
    that actually guarantees isolation, so never rely on this alone.
    """
    if not query or not query.strip():
        raise QueryRejected("Empty query.")

    try:
        statements = [s for s in sqlglot.parse(query, read="postgres") if s is not None]
    except SqlglotError as error:
        raise QueryRejected(f"Could not parse SQL: {error}") from error

    if len(statements) != 1:
        raise QueryRejected("Exactly one statement is allowed.")

    statement = statements[0]

    if not isinstance(statement, _ALLOWED_TOP_LEVEL):
        raise QueryRejected("Only SELECT queries are allowed.")

    if statement.find(*_FORBIDDEN_NODES) is not None:
        raise QueryRejected("Query contains a statement or clause that is not allowed.")

    allowed = {t.lower() for t in allowed_tables}
    cte_names = {cte.alias.lower() for cte in statement.find_all(exp.CTE)}

    for table in statement.find_all(exp.Table):
        if table.args.get("db") or table.args.get("catalog"):
            raise QueryRejected("Schema-qualified table names are not allowed.")
        if not isinstance(table.this, exp.Identifier):
            raise QueryRejected("Table-valued functions are not allowed.")
        name = table.name.lower()
        if name not in allowed and name not in cte_names:
            raise QueryRejected(f"Unknown table: {table.name}")

    for function in statement.find_all(exp.Func):
        name = _function_name(function)
        if name in _BLOCKED_FUNCTIONS or name.startswith(_BLOCKED_FUNCTION_PREFIXES):
            raise QueryRejected(f"Function not allowed: {name}")

    return query.strip().rstrip(";").strip()


def _jsonable(value):
    if isinstance(value, decimal.Decimal):
        return float(value)
    if isinstance(value, (datetime.date, datetime.datetime, datetime.time)):
        return value.isoformat()
    return value


def run_user_query(conn, user_id, query, allowed_tables, max_rows=1000, timeout_ms=10_000):
    """
    Validate and run an (AI-generated) SELECT as the user's read-only role.

    Parameters:
        conn:           Connection from get_query_connection() (autocommit).
        allowed_tables: Table names from get_user_datasets().
        max_rows:       Rows returned; the rest are never pulled into memory.
        timeout_ms:     Postgres statement timeout.

    Returns a dict: sql, columns, rows (JSON-friendly), row_count, truncated.
    Raises QueryRejected for refused SQL; psycopg errors (bad column, timeout...)
    propagate so the caller can feed the message back to the model.
    """
    cleaned = validate_select(query, allowed_tables)

    with conn.transaction():
        with conn.cursor() as setup:
            setup.execute("SET TRANSACTION READ ONLY")
            setup.execute(sql.SQL("SET LOCAL ROLE {}").format(sql.Identifier(role_name(user_id))))
            setup.execute(sql.SQL("SET LOCAL search_path = {}").format(sql.Identifier(schema_name(user_id))))
            setup.execute(sql.SQL("SET LOCAL statement_timeout = {}").format(sql.Literal(int(timeout_ms))))

        # Server-side cursor: only `max_rows + 1` rows are ever transferred.
        with conn.cursor(name="user_query") as cursor:
            cursor.execute(cleaned)
            fetched = cursor.fetchmany(max_rows + 1)
            columns = [column.name for column in cursor.description]

    truncated = len(fetched) > max_rows
    rows = [[_jsonable(v) for v in row] for row in fetched[:max_rows]]

    return {
        "sql": cleaned,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }
