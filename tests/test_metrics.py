import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baseline_diffs import comparison_rows, same_groups
from metrics import evaluate, ndcg_at_k
from train_lgbm import bootstrap_indices, per_group_metrics

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "events_clean.parquet"
EXAMPLES = ROOT / "data" / "examples.parquet"


def test_ndcg_toy():
    labels = pd.Series([1, 0, 0, 1, 0]).to_numpy()
    assert ndcg_at_k(labels, 5) == pytest.approx(0.877, abs=1e-3)


def _frame():
    # Группа 1: покупка на первом месте, NDCG = 1.
    # Группа 2: два товара и ни одной покупки. В среднее не входит.
    # Группа 3: покупка есть, но товар один. Сравнивать нечего.
    # Группа 4: покупка и два товара, но rank_group выключен. evaluate её пропускает.
    return pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 4, 4],
        "item_id": [10, 11, 20, 21, 30, 40, 41],
        "day": ["2015-08-21"] * 7,
        "label": [1, 0, 0, 0, 1, 0, 1],
        "rank_group": [True, True, True, True, True, False, False],
        "score": [0.9, 0.1, 0.8, 0.2, 0.5, 0.99, 0.01],
    })


def test_evaluate_skips_groups_without_purchase():
    res = evaluate(_frame(), "score")
    assert res["groups"] == 1
    assert res["NDCG@3"] == pytest.approx(1.0)
    assert res["NDCG@5"] == pytest.approx(1.0)
    assert res["MRR"] == pytest.approx(1.0)


def test_per_group_metrics_skips_groups_without_purchase():
    frame = _frame()
    grouped = per_group_metrics(frame, frame["score"].to_numpy())
    # Группы 1 и 4: две строки и покупка. Группы 2 и 3 не входят.
    assert len(grouped["ndcg3"]) == 2
    assert grouped["ndcg3"][0] == pytest.approx(1.0)
    assert grouped["ndcg5"][0] == pytest.approx(1.0)
    assert grouped["mrr"][0] == pytest.approx(1.0)
    assert grouped["ndcg3"].mean() > 0.5
    assert list(grouped["user_id"]) == [1, 4]


def test_baseline_diffs_use_the_same_groups():
    frame = _frame()
    left = per_group_metrics(frame, frame["score"].to_numpy(), seed=42)
    right = per_group_metrics(frame, np.zeros(len(frame)), seed=42)
    grouped = {"A": left, "popularity": right}
    assert same_groups(grouped) == 2
    indices = bootstrap_indices(2, 1000, 42)
    table = comparison_rows(
        {"A": left["ndcg3"], "popularity": right["ndcg3"], "B": left["ndcg3"], "C": left["ndcg3"], "prior_views": right["ndcg3"]},
        indices,
    )
    # Одни и те же индексы: A−popularity и B−popularity совпадают, раз массив A равен B.
    a_pop = table.loc[table["comparison"] == "A-popularity"].iloc[0]
    b_pop = table.loc[table["comparison"] == "B-popularity"].iloc[0]
    assert a_pop["n_groups"] == 2
    assert a_pop["ndcg_3"] == pytest.approx(b_pop["ndcg_3"])
    assert a_pop["ndcg_3_low"] == pytest.approx(b_pop["ndcg_3_low"])
    assert a_pop["ndcg_3_high"] == pytest.approx(b_pop["ndcg_3_high"])
    assert a_pop["ndcg_3"] == pytest.approx(left["ndcg3"].mean() - right["ndcg3"].mean())


@pytest.mark.needs_data(
    "data/events_clean.parquet",
    "data/examples.parquet",
)
def test_label_is_transaction_only():
    con = duckdb.connect()
    n_bad, n_cart_only_positive = con.sql(f"""
        WITH day_events AS (
            SELECT user_id, item_id, day,
                   max(event = 'transaction') AS has_buy,
                   max(event = 'addtocart') AS has_cart
            FROM '{EVENTS}'
            GROUP BY user_id, item_id, day
        ),
        joined AS (
            SELECT e.label, d.has_buy, d.has_cart
            FROM '{EXAMPLES}' e
            JOIN day_events d USING (user_id, item_id, day)
            WHERE e.split IN ('train', 'val')
        )
        SELECT
            (SELECT count(*) FROM joined
             WHERE label <> CASE WHEN has_buy THEN 1 ELSE 0 END),
            (SELECT count(*) FROM joined
             WHERE has_cart AND NOT has_buy AND label <> 0)
    """).fetchone()
    assert n_bad == 0
    assert n_cart_only_positive == 0

    inconsistent = con.sql(f"""
        WITH groups AS (
            SELECT user_id, day,
                   count(*) AS n_items,
                   sum(label) AS n_pos,
                   min(rank_group::INT) AS rank_min,
                   max(rank_group::INT) AS rank_max
            FROM '{EXAMPLES}'
            WHERE split IN ('train', 'val')
            GROUP BY user_id, day
        )
        SELECT count(*) FROM groups
        WHERE rank_min <> rank_max
           OR rank_max <> CASE WHEN n_items >= 2 AND n_pos >= 1 THEN 1 ELSE 0 END
    """).fetchone()[0]
    assert inconsistent == 0
