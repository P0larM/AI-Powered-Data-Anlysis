import numpy as np
import pandas as pd
import pytest

from app.database.ingest import pg_type_for, prepare_for_load, safe_identifiers
from app.database.tenancy import role_name, schema_name


def test_schema_and_role_names():
    assert schema_name("demo") == "user_demo"
    assert role_name("demo") == "user_demo_ro"


@pytest.mark.parametrize("bad", ["Demo", "a-b", "x; DROP SCHEMA public", "", "a" * 41])
def test_bad_user_ids_are_rejected(bad):
    with pytest.raises(ValueError):
        schema_name(bad)


def test_safe_identifiers_dedupe_and_truncate():
    names = safe_identifiers(["Order ID", "order-id", "x" * 100])
    assert names[0] == "order_id" and names[1] == "order_id_2"
    assert len(names[2]) == 63


def test_pg_types():
    df = pd.DataFrame({
        "i": [1, 2], "f": [1.5, 2.5], "b": [True, False],
        "t": ["a", "b"], "d": pd.to_datetime(["2020-01-01", "2020-01-02"]),
    })
    assert [pg_type_for(df[c]) for c in df.columns] == [
        "BIGINT", "DOUBLE PRECISION", "BOOLEAN", "TEXT", "TIMESTAMP"]


def test_whole_number_floats_become_integers_but_real_floats_do_not():
    df = pd.DataFrame({"age": [39.0, np.nan, 50.0], "score": [1.5, np.nan, 2.5]})
    out = prepare_for_load(df)
    assert pg_type_for(out["age"]) == "BIGINT"
    assert pg_type_for(out["score"]) == "DOUBLE PRECISION"
    assert out["age"].isna().sum() == 1
