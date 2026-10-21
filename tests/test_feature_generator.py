import sys
from pathlib import Path

import duckdb
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from feature_selector import decide, roc_auc_small, summarize_group_auc

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "events_clean.parquet"
CANDIDATES = ROOT / "data" / "feature_candidates.parquet"
EXAMPLES = ROOT / "data" / "examples.parquet"
CATEGORIES = ROOT / "data" / "item_category.parquet"
# Сутки начинаются в 03:00 UTC, та же граница, что в prepare.py.
DAY_SHIFT_MS = 3 * 3600 * 1000
SEED = 0


def _con():
    return duckdb.connect()


@pytest.mark.needs_data(
    "data/feature_candidates.parquet",
    "data/examples.parquet",
    "data/events_clean.parquet",
)
def test_features_ignore_events_on_day_d():
    con = _con()
    pool = con.sql(f"""
        SELECT c.user_id, c.item_id, c.day
        FROM '{CANDIDATES}' c
        JOIN '{EXAMPLES}' e USING (user_id, item_id, day)
        WHERE e.split <> 'test'
          AND NOT e.category_fallback
          AND e.category_id IS NOT NULL
          AND c.user_item__view__7__count > 0
    """).df()
    rng = np.random.default_rng(SEED)
    sample = pool.iloc[rng.choice(len(pool), size=5, replace=False)].reset_index(drop=True)
    con.register("sample", sample)
    check = con.sql(f"""
        WITH expected AS (
            SELECT s.user_id, s.item_id, s.day,
                (SELECT count(*) FROM '{EVENTS}' ev
                 WHERE ev.user_id = s.user_id AND ev.event = 'view'
                   AND ev.day < s.day AND ev.day >= s.day - INTERVAL 7 DAY) AS user_view_7,
                (SELECT count(*) FROM '{EVENTS}' ev
                 WHERE ev.item_id = s.item_id AND ev.event = 'transaction'
                   AND ev.day < s.day AND ev.day >= s.day - INTERVAL 30 DAY) AS item_buy_30,
                (SELECT count(*) FROM '{EVENTS}' ev
                 WHERE ev.user_id = s.user_id AND ev.item_id = s.item_id AND ev.event = 'view'
                   AND ev.day < s.day AND ev.day >= s.day - INTERVAL 1 DAY) AS ui_view_1,
                (SELECT count(DISTINCT ev.item_id) FROM '{EVENTS}' ev
                 WHERE ev.user_id = s.user_id AND ev.event = 'view'
                   AND ev.day < s.day AND ev.day >= s.day - INTERVAL 7 DAY) AS user_items_7,
                (SELECT date_diff('day', max(ev.day), s.day) FROM '{EVENTS}' ev
                 WHERE ev.user_id = s.user_id AND ev.event = 'view' AND ev.day < s.day)
                    AS user_view_recency
            FROM sample s
        )
        SELECT count(*) FROM expected e
        JOIN '{CANDIDATES}' c USING (user_id, item_id, day)
        WHERE c.user__view__7__count IS DISTINCT FROM e.user_view_7
           OR c.item__transaction__30__count IS DISTINCT FROM e.item_buy_30
           OR c.user_item__view__1__count IS DISTINCT FROM e.ui_view_1
           OR c.user__view__7__nunique IS DISTINCT FROM e.user_items_7
           OR c.user__view__all__recency IS DISTINCT FROM e.user_view_recency
    """).fetchone()[0]
    assert check == 0


@pytest.mark.needs_data(
    "data/feature_candidates.parquet",
    "data/examples.parquet",
    "data/events_clean.parquet",
)
def test_pair_with_events_only_on_day_d_has_empty_pair_features():
    con = _con()
    n_pairs, n_bad = con.sql(f"""
        WITH pair_days AS (
            SELECT user_id, item_id, min(day) AS only_day
            FROM '{EVENTS}'
            GROUP BY user_id, item_id
            HAVING count(DISTINCT day) = 1
        ),
        only_today AS (
            SELECT c.user_item__view__1__count, c.user_item__view__7__count,
                   c.user_item__view__30__count, c.user_item__addtocart__1__count,
                   c.user_item__transaction__1__count, c.user_item__view__1__share,
                   c.user_item__view__7_30__trend, c.user_item__view__all__recency,
                   c.user_item__transaction__all__recency
            FROM '{CANDIDATES}' c
            JOIN pair_days p
              ON p.user_id = c.user_id AND p.item_id = c.item_id AND p.only_day = c.day
            JOIN '{EXAMPLES}' e
              ON e.user_id = c.user_id AND e.item_id = c.item_id AND e.day = c.day
            WHERE e.split <> 'test'
        )
        SELECT
            (SELECT count(*) FROM only_today),
            (SELECT count(*) FROM only_today
             WHERE user_item__view__1__count <> 0
                OR user_item__view__7__count <> 0
                OR user_item__view__30__count <> 0
                OR user_item__addtocart__1__count <> 0
                OR user_item__transaction__1__count <> 0
                OR user_item__view__1__share IS NOT NULL
                OR user_item__view__7_30__trend IS NOT NULL
                OR user_item__view__all__recency IS NOT NULL
                OR user_item__transaction__all__recency IS NOT NULL)
    """).fetchone()
    assert n_pairs > 0
    assert n_bad == 0


@pytest.mark.needs_data(
    "data/feature_candidates.parquet",
    "data/examples.parquet",
)
def test_fallback_category_features_are_null():
    con = _con()
    n_bad = con.sql(f"""
        SELECT count(*)
        FROM '{CANDIDATES}' c
        JOIN '{EXAMPLES}' e USING (user_id, item_id, day)
        WHERE e.split <> 'test'
          AND (e.category_fallback OR e.category_id IS NULL)
          AND (
              c.category__view__7__count IS NOT NULL
              OR c.category__transaction__30__share IS NOT NULL
              OR c.category__view__all__recency IS NOT NULL
              OR c.category__view__7__nunique IS NOT NULL
              OR c.category__addtocart__7_30__trend IS NOT NULL
          )
    """).fetchone()[0]
    assert n_bad == 0


@pytest.mark.needs_data(
    "data/feature_candidates.parquet",
    "data/examples.parquet",
    "data/item_category.parquet",
)
def test_category_change_uses_only_records_before_day_d():
    con = _con()
    pool = con.sql(f"""
        SELECT c.user_id, c.item_id, c.day, c.item__category__30__changed
        FROM '{CANDIDATES}' c
        JOIN '{EXAMPLES}' e USING (user_id, item_id, day)
        WHERE e.split <> 'test' AND c.item__category__30__changed = 1
    """).df()
    rng = np.random.default_rng(SEED)
    sample = pool.iloc[rng.choice(len(pool), size=5, replace=False)].reset_index(drop=True)
    con.register("sample", sample[["user_id", "item_id", "day"]])
    expected = con.sql(f"""
        WITH bounds AS (
            SELECT item_id, day,
                   epoch_ms(CAST(day AS TIMESTAMP)) + {DAY_SHIFT_MS} AS day_start,
                   epoch_ms(CAST(day AS TIMESTAMP) - INTERVAL 30 DAY) + {DAY_SHIFT_MS} AS win_start
            FROM sample
        ),
        item_cat AS (
            SELECT item_id, ts, category_id
            FROM '{CATEGORIES}'
            WHERE category_id IS NOT NULL
            ORDER BY item_id, ts
        ),
        vals AS (
            SELECT b.item_id, b.day, c.category_id
            FROM bounds b
            JOIN item_cat c
              ON c.item_id = b.item_id AND c.ts >= b.win_start AND c.ts < b.day_start
            UNION ALL
            SELECT b.item_id, b.day, prev.category_id
            FROM bounds b
            ASOF LEFT JOIN item_cat prev
              ON b.item_id = prev.item_id AND b.win_start > prev.ts
            WHERE prev.category_id IS NOT NULL
        )
        SELECT b.item_id, b.day,
               coalesce((
                   SELECT (count(DISTINCT category_id) >= 2)::DOUBLE
                   FROM vals v
                   WHERE v.item_id = b.item_id AND v.day = b.day
               ), 0) AS changed
        FROM bounds b
    """).df()
    got = sample.merge(expected, on=["item_id", "day"])
    assert (got["item__category__30__changed"] == got["changed"]).all()


def test_selector_drops_weak_and_duplicate_features():
    # group_auc: 1, 1, 0.75, 0.5. Силы 0.5, 0.5, 0.25, 0, среднее 0.3125.
    columns = {
        "a_strong": np.array([0.0, 0.0, 1.0, 1.0]),
        "b_copy": np.array([0.0, 0.0, 1.0, 1.0]),
        "mid": np.array([0.0, 0.0, 0.0, 1.0]),
        "weak": np.array([0.0, 1.0, 0.0, 1.0]),
    }
    table, threshold = decide(
        list(columns),
        [0.9, 0.9, 0.7, 0.4],
        [1.0, 1.0, 0.75, 0.5],
        [2, 2, 1, 0],
        [0, 0, 0, 1],
        columns,
    )
    assert threshold == 0.3125
    kept = set(table.loc[table["прошёл"] == 1, "признак"])
    assert kept == {"a_strong"}
    reasons = dict(zip(table["признак"], table["причина"]))
    assert reasons["b_copy"] == "дубль: a_strong"
    assert reasons["mid"] == "ниже порога"
    assert reasons["weak"] == "ниже порога"
    assert reasons["a_strong"] == "прошёл"
    assert table.loc[table["признак"] == "a_strong", "global_auc"].iloc[0] == 0.9
    assert table.loc[table["признак"] == "weak", "constant_in_group"].iloc[0] == 1


def test_constant_within_group_has_group_auc_half():
    y = np.array([0, 1, 0, 1, 0, 0])
    starts = np.array([0, 2, 4])
    ends = np.array([2, 4, 6])
    # В каждой группе значение одно и то же, хотя между группами оно разное.
    constant = np.array([3.0, 3.0, 9.0, 9.0, 1.0, 1.0])
    group_auc, n_used, is_constant = summarize_group_auc(y, constant, starts, ends)
    assert is_constant
    assert group_auc == 0.5
    assert n_used == 0

    # Третья группа вся с меткой 0, её пропускаем. В первых двух положительный выше.
    varying = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 5.0])
    group_auc, n_used, is_constant = summarize_group_auc(y, varying, starts, ends)
    assert not is_constant
    assert n_used == 2
    assert group_auc == 1.0

    tied = np.array([0.2, 0.2, 0.9])
    labels = np.array([1, 0, 1])
    assert abs(roc_auc_small(labels, tied) - roc_auc_score(labels, tied)) < 1e-12


@pytest.mark.needs_data("data/feature_candidates.parquet")
def test_default_windows_keep_the_frozen_152_names():
    from feature_generator import SEARCH_WINDOWS, WINDOWS, candidate_feature_names

    described = [row[0] for row in duckdb.sql(f"DESCRIBE SELECT * FROM '{CANDIDATES}'").fetchall()]
    frozen = [name for name in described if name not in ("user_id", "item_id", "day")]
    assert candidate_feature_names() == frozen
    assert candidate_feature_names(WINDOWS) == frozen
    assert len(frozen) == 152
    searched = candidate_feature_names(SEARCH_WINDOWS)
    assert len(searched) == 272
    assert set(frozen).issubset(searched)
