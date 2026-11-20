from app.database.connection import get_query_connection


def attempt(label, statements):
    """Run statements in one transaction as the user's role, skipping the validator."""
    with get_query_connection() as conn:
        try:
            with conn.transaction():
                conn.execute('SET LOCAL ROLE "user_demo_ro"')
                conn.execute("SET LOCAL search_path = user_demo")
                for statement in statements:
                    result = conn.execute(statement)
                    if result.description:
                        print(label, "->", result.fetchone())
        except Exception as error:
            print(label, "-> BLOCKED:", type(error).__name__)


attempt("read own table  ", ["SELECT COUNT(*) FROM adult"])
attempt("read app_meta   ", ["SELECT * FROM app_meta.datasets"])
attempt("delete own rows ", ["DELETE FROM adult"])
attempt("drop own table  ", ["DROP TABLE adult"])