import pytest

from app.database.queries import QueryRejected, validate_select

TABLES = ["adult", "sales"]


@pytest.mark.parametrize("query", [
    "SELECT * FROM adult",
    "SELECT sex, COUNT(*) FROM adult WHERE age > 30 GROUP BY sex ORDER BY 2 DESC LIMIT 5;",
    "WITH r AS (SELECT * FROM adult WHERE income = '>50K') SELECT COUNT(*) FROM r",
    "SELECT a.age FROM adult a JOIN sales s ON s.id = a.age",
    "SELECT age FROM adult UNION SELECT age FROM adult",
    "SELECT * FROM adult WHERE workclass LIKE '%gov%'",
])
def test_good_queries_pass(query):
    assert validate_select(query, TABLES)


@pytest.mark.parametrize("query", [
    "DROP TABLE adult",
    "DELETE FROM adult",
    "INSERT INTO adult VALUES (1)",
    "UPDATE adult SET age = 1",
    "SELECT * FROM adult; DROP TABLE adult",
    "SELECT * FROM user_other.adult",
    "SELECT * FROM other_db.public.adult",
    "SELECT * FROM secrets",
    "SELECT * FROM app_meta.datasets",
    "SELECT pg_sleep(60)",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT set_config('role', 'postgres', false)",
    "SELECT * INTO stolen FROM adult",
    "SELECT * FROM adult FOR UPDATE",
    "WITH d AS (DELETE FROM adult RETURNING *) SELECT * FROM d",
    "SELECT * FROM generate_series(1, 10)",
    "",
    "   ",
])
def test_bad_queries_are_rejected(query):
    with pytest.raises(QueryRejected):
        validate_select(query, TABLES)
