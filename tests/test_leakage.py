import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import ndcg_at_k

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "events_clean.parquet"
FEATURES = ROOT / "data" / "features_manual.parquet"
SEED = 0
# То же значение, что в features_manual.py: не было просмотра — 999, не NULL.
RECENCY_MISSING = 999


def test_ndcg_toy():
    labels = np.array([1, 0, 0, 1, 0])
    assert ndcg_at_k(labels, 5) == pytest.approx(0.877, abs=1e-3)


def _sample(con, split, n, nonzero_pair):
    where = f"split = '{split}'"
    if nonzero_pair:
        where += " AND (ui_views_1d > 0 OR ui_views_7d > 0 OR ui_views_30d > 0 OR ui_cart_before = 1)"
    df = con.sql(f"""
        SELECT user_id, item_id, day
        FROM '{FEATURES}'
        WHERE {where}
    """).df()
    rng = np.random.default_rng(SEED)
    take = rng.choice(len(df), size=n, replace=False)
    return df.iloc[take].reset_index(drop=True)


def test_features_match_events():
    con = duckdb.connect()
    sample = pd.concat(
        [_sample(con, "val", 5, True), _sample(con, "train", 5, False)],
        ignore_index=True,
    )
    con.register("sample", sample)
    check = con.sql(f"""
        WITH expected AS (
            SELECT s.user_id, s.item_id, s.day,
                (SELECT count(*) FROM '{EVENTS}' e
                 WHERE e.user_id = s.user_id AND e.event = 'view'
                   AND e.day < s.day AND e.day >= s.day - INTERVAL 7 DAY) AS user_views_7d,
                (SELECT count(*) FROM '{EVENTS}' e
                 WHERE e.item_id = s.item_id AND e.event = 'transaction'
                   AND e.day < s.day AND e.day >= s.day - INTERVAL 30 DAY) AS item_buys_30d,
                (SELECT count(*) FROM '{EVENTS}' e
                 WHERE e.user_id = s.user_id AND e.item_id = s.item_id AND e.event = 'view'
                   AND e.day < s.day AND e.day >= s.day - INTERVAL 7 DAY) AS ui_views_7d,
                coalesce(
                    (SELECT date_diff('day', max(e.day), s.day) FROM '{EVENTS}' e
                     WHERE e.user_id = s.user_id AND e.item_id = s.item_id
                       AND e.event = 'view' AND e.day < s.day),
                    {RECENCY_MISSING}) AS ui_days_since_view,
                (SELECT (count(*) > 0)::INTEGER FROM '{EVENTS}' e
                 WHERE e.user_id = s.user_id AND e.item_id = s.item_id
                   AND e.event = 'addtocart' AND e.day < s.day) AS ui_cart_before
            FROM sample s
        )
        SELECT count(*) AS n_bad
        FROM expected e
        JOIN '{FEATURES}' f USING (user_id, item_id, day)
        WHERE f.user_views_7d IS DISTINCT FROM e.user_views_7d
           OR f.item_buys_30d IS DISTINCT FROM e.item_buys_30d
           OR f.ui_views_7d IS DISTINCT FROM e.ui_views_7d
           OR f.ui_days_since_view IS DISTINCT FROM e.ui_days_since_view
           OR f.ui_cart_before IS DISTINCT FROM e.ui_cart_before
    """).fetchone()[0]
    assert len(sample) == 10
    assert check == 0


def test_pair_features_ignore_same_day_events():
    """У пары, чьи события только в день D, эти события не входят в признаки."""
    con = duckdb.connect()
    n_pairs, n_bad = con.sql(f"""
        WITH pair_days AS (
            SELECT user_id, item_id, min(day) AS only_day
            FROM '{EVENTS}'
            GROUP BY user_id, item_id
            HAVING count(DISTINCT day) = 1
        ),
        only_today AS (
            SELECT f.user_id, f.item_id, f.day
            FROM '{FEATURES}' f
            JOIN pair_days p
              ON p.user_id = f.user_id AND p.item_id = f.item_id AND p.only_day = f.day
            WHERE f.split <> 'test'
        )
        SELECT
            (SELECT count(*) FROM only_today),
            (SELECT count(*) FROM only_today o
             JOIN '{FEATURES}' f USING (user_id, item_id, day)
             WHERE f.ui_views_1d <> 0 OR f.ui_views_7d <> 0 OR f.ui_views_30d <> 0
                OR f.ui_cart_before <> 0
                OR f.ui_days_since_view <> {RECENCY_MISSING})
    """).fetchone()
    assert n_pairs > 0
    assert n_bad == 0
