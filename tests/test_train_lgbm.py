import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_lgbm import EXAMPLES_PATH, TRAIN_LAST, VAL_FIRST, VAL_LAST, ensure_train_rows, load_keys


def test_training_rows_exclude_val_and_test():
    keys = load_keys()
    assert set(keys["split"]) == {"train", "val"}
    train = keys[keys["split"] == "train"]
    val = keys[keys["split"] == "val"]
    assert pd.to_datetime(train["day"]).max() <= TRAIN_LAST
    assert pd.to_datetime(val["day"]).min() >= VAL_FIRST
    assert pd.to_datetime(val["day"]).max() <= VAL_LAST
    overlap = train.merge(val, on=["user_id", "item_id", "day"])
    assert overlap.empty
    ensure_train_rows(train)

    test_rows = duckdb.sql(f"""
        SELECT count(*) FROM '{EXAMPLES_PATH}' WHERE split = 'test'
    """).fetchone()[0]
    assert test_rows > 0
    leaked = keys.merge(
        duckdb.sql(f"""
            SELECT user_id, item_id, day
            FROM '{EXAMPLES_PATH}'
            WHERE split = 'test'
        """).df(),
        on=["user_id", "item_id", "day"],
    )
    assert leaked.empty

    bad_split = train.copy()
    bad_split.loc[bad_split.index[0], "split"] = "test"
    with pytest.raises(RuntimeError):
        ensure_train_rows(bad_split)

    bad_day = train.copy()
    bad_day.loc[bad_day.index[0], "day"] = pd.Timestamp("2015-09-04")
    with pytest.raises(RuntimeError):
        ensure_train_rows(bad_day)
