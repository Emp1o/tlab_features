import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from feature_generator import (
    CATEGORY_ALL_BUY_COUNT,
    CATEGORY_ALL_REPEAT_SHARE,
    REPEAT_RAW_MIN,
    ITEM_AVAILABLE,
    ITEM_AVAILABLE_SWITCHES,
    USER_CATEGORY_BUY_COUNT,
    USER_CATEGORY_BUY_RECENCY,
    USER_CATEGORY_BUY_SHARE,
)
from metrics import ndcg_at_k

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "events_clean.parquet"
FEATURES = ROOT / "data" / "features_manual.parquet"
EXAMPLES = ROOT / "data" / "examples.parquet"
CANDIDATES = ROOT / "data" / "feature_candidates.parquet"
CATEGORIES = ROOT / "data" / "item_category.parquet"
DAY_SHIFT_MS = 3 * 3600 * 1000
# Последний день val. Покупки test-периода в окна train/val не входят.
VAL_LAST_DAY = "2015-09-03"
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


@pytest.mark.needs_data(
    "data/features_manual.parquet",
    "data/events_clean.parquet",
)
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


@pytest.mark.needs_data(
    "data/features_manual.parquet",
    "data/events_clean.parquet",
)
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


@pytest.mark.needs_data(
    "data/item_category.parquet",
    "data/events_clean.parquet",
    "data/examples.parquet",
    "data/feature_candidates.parquet",
)
def test_user_category_buys_do_not_leak():
    """Покупки в категории не видят день D, будущее и категорию с начала этого дня.

    Категория строки — строго до 03:00 дня D. Категория прошлой покупки — на момент
    покупки. Нет категории — NULL. Ноль покупок пользователя за 30 дней — доля NULL.
    """
    con = duckdb.connect()
    con.sql(f"""
        CREATE TABLE item_cat AS
        SELECT item_id, ts, category_id
        FROM '{CATEGORIES}'
        WHERE category_id IS NOT NULL
        ORDER BY item_id, ts
    """)
    con.sql(f"""
        CREATE TABLE buys AS
        SELECT user_id, item_id, ts, day,
               epoch_ms(CAST(day AS TIMESTAMP)) + {DAY_SHIFT_MS} AS day_start
        FROM '{EVENTS}'
        WHERE event = 'transaction' AND day <= DATE '{VAL_LAST_DAY}'
    """)
    con.sql("""
        CREATE TABLE buy_at_purchase AS
        SELECT b.user_id, b.day, c.category_id
        FROM buys b
        ASOF LEFT JOIN item_cat c ON b.item_id = c.item_id AND b.ts >= c.ts
        WHERE c.category_id IS NOT NULL
    """)
    # Снимок ровно в 03:00 дня покупки здесь отброшен: это не «на момент покупки».
    con.sql("""
        CREATE TABLE buy_before_day AS
        SELECT b.user_id, b.day, c.category_id
        FROM buys b
        ASOF LEFT JOIN item_cat c ON b.item_id = c.item_id AND b.day_start > c.ts
        WHERE c.category_id IS NOT NULL
    """)
    con.sql(f"""
        CREATE TABLE rows AS
        SELECT e.user_id, e.item_id, e.day,
               epoch_ms(CAST(e.day AS TIMESTAMP)) + {DAY_SHIFT_MS} AS day_start,
               c.{USER_CATEGORY_BUY_COUNT} AS got_count,
               c.{USER_CATEGORY_BUY_SHARE} AS got_share,
               c.user__transaction__30__count AS got_user_buys
        FROM '{EXAMPLES}' e
        JOIN '{CANDIDATES}' c USING (user_id, item_id, day)
        WHERE e.split IN ('train', 'val')
    """)
    con.sql("""
        CREATE TABLE row_strict AS
        SELECT r.user_id, r.item_id, r.day, r.day_start, r.got_count, r.got_share,
               r.got_user_buys, c.category_id
        FROM rows r
        ASOF LEFT JOIN item_cat c
          ON r.item_id = c.item_id AND r.day_start > c.ts
    """)
    con.sql("""
        CREATE TABLE row_inclusive AS
        SELECT r.user_id, r.item_id, r.day, c.category_id
        FROM rows r
        ASOF LEFT JOIN item_cat c
          ON r.item_id = c.item_id AND r.day_start >= c.ts
    """)

    def window_count(buy_table, category_table):
        con.sql(f"""
            CREATE OR REPLACE TABLE {category_table}_sum AS
            SELECT r.user_id, r.item_id, r.day,
                   CASE WHEN r.category_id IS NULL THEN NULL
                        ELSE coalesce(sum(b.n), 0) END AS n
            FROM {category_table} r
            LEFT JOIN (
                SELECT user_id, category_id, day, count(*) AS n
                FROM {buy_table}
                GROUP BY 1, 2, 3
            ) b
              ON r.category_id IS NOT NULL
             AND b.user_id = r.user_id
             AND b.category_id = r.category_id
             AND b.day < r.day
             AND b.day >= r.day - INTERVAL 30 DAY
            GROUP BY r.user_id, r.item_id, r.day, r.category_id
        """)

    window_count("buy_at_purchase", "row_strict")
    # Та же категория строки, но у покупки снят снимок ровно в 03:00 её дня.
    con.sql("""
        CREATE TABLE row_for_boundary AS
        SELECT user_id, item_id, day, category_id FROM row_strict
    """)
    window_count("buy_before_day", "row_for_boundary")
    con.sql("""
        CREATE TABLE user_buys AS
        SELECT r.user_id, r.item_id, r.day, coalesce(sum(b.n), 0) AS n
        FROM rows r
        LEFT JOIN (
            SELECT user_id, day, count(*) AS n
            FROM buys
            GROUP BY 1, 2
        ) b
          ON b.user_id = r.user_id
         AND b.day < r.day
         AND b.day >= r.day - INTERVAL 30 DAY
        GROUP BY r.user_id, r.item_id, r.day
    """)
    stats = con.sql("""
        WITH cmp AS (
            SELECT s.got_count, s.got_share, s.category_id, s.day,
                   s.user_id, e.n AS exp_count, u.n AS user_buys,
                   i.category_id AS inclusive_category,
                   p.n AS boundary_count
            FROM row_strict s
            JOIN row_strict_sum e USING (user_id, item_id, day)
            JOIN user_buys u USING (user_id, item_id, day)
            JOIN row_inclusive i USING (user_id, item_id, day)
            JOIN row_for_boundary_sum p USING (user_id, item_id, day)
        ),
        same_day AS (
            SELECT user_id, category_id, day, count(*) AS n
            FROM buy_at_purchase
            GROUP BY 1, 2, 3
        )
        SELECT
            (SELECT count(*) FROM cmp
             WHERE got_count IS DISTINCT FROM exp_count
                OR (category_id IS NULL OR user_buys = 0) AND got_share IS NOT NULL
                OR category_id IS NOT NULL AND user_buys > 0
                   AND (got_share IS NULL OR abs(got_share - exp_count / user_buys) > 1e-9)),
            (SELECT count(*) FROM cmp c
             JOIN same_day d
               ON d.user_id = c.user_id AND d.category_id = c.category_id AND d.day = c.day),
            (SELECT count(*) FROM cmp
             WHERE category_id IS DISTINCT FROM inclusive_category),
            (SELECT count(*) FROM cmp WHERE exp_count IS DISTINCT FROM boundary_count),
            (SELECT count(*) FROM cmp
             WHERE user_buys = 0 AND (got_share = 0 OR got_share IS NOT NULL)),
            (SELECT count(*) FROM cmp
             WHERE category_id IS NULL AND (got_count IS NOT NULL OR got_share IS NOT NULL))
    """).fetchone()
    n_bad, n_same_day, n_strict, n_purchase_time, n_zero_share, n_unknown = stats
    assert n_bad == 0
    assert n_same_day > 0
    assert n_strict > 0
    assert n_purchase_time > 0
    assert n_zero_share == 0
    assert n_unknown == 0


@pytest.mark.needs_data(
    "data/item_category.parquet",
    "data/events_clean.parquet",
    "data/examples.parquet",
    "data/feature_candidates.parquet",
)
def test_category_all_purchase_features_do_not_leak():
    """Категорийные признаки считают покупки всех пользователей до дня D.

    Повторная покупка — у покупателя уже была покупка в этой категории раньше
    по timestamp, не в ту же миллисекунду. Меньше 5 покупок сглаживаются приором.
    """
    con = duckdb.connect()
    con.sql(f"""
        CREATE TABLE item_cat AS
        SELECT item_id, ts, category_id
        FROM '{CATEGORIES}'
        WHERE category_id IS NOT NULL
        ORDER BY item_id, ts
    """)
    con.sql(f"""
        CREATE TABLE buys AS
        SELECT user_id, item_id, ts, day
        FROM '{EVENTS}'
        WHERE event = 'transaction' AND day <= DATE '{VAL_LAST_DAY}'
    """)
    con.sql("""
        CREATE TABLE buy_at_purchase AS
        SELECT b.user_id, b.ts, b.day, c.category_id,
               count(*) OVER (
                   PARTITION BY b.user_id, c.category_id
                   ORDER BY b.ts
                   RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
               ) > 0 AS is_repeat
        FROM buys b
        ASOF LEFT JOIN item_cat c ON b.item_id = c.item_id AND b.ts >= c.ts
        WHERE c.category_id IS NOT NULL
    """)
    # Окно не должно считать покупку с тем же timestamp предыдущей.
    repeat_bad = con.sql("""
        SELECT count(*) FROM buy_at_purchase b
        WHERE is_repeat <> EXISTS (
            SELECT 1 FROM buy_at_purchase p
            WHERE p.user_id = b.user_id
              AND p.category_id = b.category_id
              AND p.ts < b.ts
        )
    """).fetchone()[0]
    assert repeat_bad == 0
    con.sql(f"""
        CREATE TABLE rows AS
        SELECT e.user_id, e.item_id, e.day,
               epoch_ms(CAST(e.day AS TIMESTAMP)) + {DAY_SHIFT_MS} AS day_start,
               c.{CATEGORY_ALL_BUY_COUNT} AS got_count,
               c.{CATEGORY_ALL_REPEAT_SHARE} AS got_share
        FROM '{EXAMPLES}' e
        JOIN '{CANDIDATES}' c USING (user_id, item_id, day)
        WHERE e.split IN ('train', 'val')
    """)
    con.sql("""
        CREATE TABLE row_strict AS
        SELECT r.*, c.category_id
        FROM rows r
        ASOF LEFT JOIN item_cat c
          ON r.item_id = c.item_id AND r.day_start > c.ts
    """)
    con.sql("""
        CREATE TABLE row_inclusive AS
        SELECT r.user_id, r.item_id, r.day, c.category_id
        FROM rows r
        ASOF LEFT JOIN item_cat c
          ON r.item_id = c.item_id AND r.day_start >= c.ts
    """)
    stats = con.sql(f"""
        WITH daily AS (
            SELECT category_id, day,
                   count(*) AS n,
                   count(*) FILTER (WHERE is_repeat) AS n_repeat
            FROM buy_at_purchase
            GROUP BY 1, 2
        ),
        expected AS (
            SELECT r.user_id, r.item_id, r.day, r.category_id,
                   r.got_count, r.got_share,
                   CASE WHEN r.category_id IS NULL THEN NULL
                        ELSE coalesce(sum(d.n) FILTER (
                            WHERE d.day >= r.day - INTERVAL 30 DAY
                        ), 0) END AS exp_count,
                   CASE WHEN r.category_id IS NULL THEN NULL
                        ELSE coalesce(sum(d.n), 0) END AS hist_n,
                   CASE WHEN r.category_id IS NULL THEN NULL
                        ELSE coalesce(sum(d.n_repeat), 0) END AS hist_repeat
            FROM row_strict r
            LEFT JOIN daily d
              ON r.category_id IS NOT NULL
             AND d.category_id = r.category_id
             AND d.day < r.day
            GROUP BY r.user_id, r.item_id, r.day, r.category_id, r.got_count, r.got_share
        ),
        scored AS (
            SELECT e.*, i.category_id AS inclusive_category,
                   CASE WHEN e.hist_n IS NULL OR e.hist_n = 0 THEN NULL
                        WHEN e.hist_n < {REPEAT_RAW_MIN}
                        THEN (e.hist_repeat + 1)::DOUBLE / (e.hist_n + 2)
                        ELSE e.hist_repeat::DOUBLE / e.hist_n
                   END AS exp_share
            FROM expected e
            JOIN row_inclusive i USING (user_id, item_id, day)
        ),
        same_day AS (
            SELECT category_id, day FROM buy_at_purchase GROUP BY 1, 2
        )
        SELECT
            (SELECT count(*) FROM scored
             WHERE got_count IS DISTINCT FROM exp_count
                OR got_share IS DISTINCT FROM exp_share
                   AND NOT (got_share IS NOT NULL AND exp_share IS NOT NULL
                            AND abs(got_share - exp_share) <= 1e-9)),
            (SELECT count(*) FROM scored s
             JOIN same_day d ON d.category_id = s.category_id AND d.day = s.day),
            (SELECT count(*) FROM scored
             WHERE category_id IS DISTINCT FROM inclusive_category),
            (SELECT count(*) FROM scored WHERE hist_n = 0 AND got_share IS NOT NULL),
            (SELECT count(*) FROM scored WHERE category_id IS NULL AND got_count IS NOT NULL),
            (SELECT count(*) FROM scored WHERE hist_n > 0 AND hist_n < {REPEAT_RAW_MIN}),
            (SELECT count(*) FROM scored WHERE hist_n >= {REPEAT_RAW_MIN})
    """).fetchone()
    n_bad, n_same_day, n_strict, n_empty_share, n_unknown, n_smoothed, n_raw = stats
    assert n_bad == 0
    assert n_same_day > 0
    assert n_strict > 0
    assert n_empty_share == 0
    assert n_unknown == 0
    assert n_smoothed > 0
    assert n_raw > 0


@pytest.mark.needs_data(
    "data/item_category.parquet",
    "data/events_clean.parquet",
    "data/examples.parquet",
    "data/feature_candidates.parquet",
)
def test_user_category_recency_does_not_leak():
    """Давность — дни до последней покупки пользователя в категории, строго до дня D.

    Категория строки известна только до 03:00. Категория покупки — на момент покупки.
    Покупка в день группы не делает давность нулевой.
    """
    con = duckdb.connect()
    con.sql(f"""
        CREATE TABLE item_cat AS
        SELECT item_id, ts, category_id
        FROM '{CATEGORIES}'
        WHERE category_id IS NOT NULL
        ORDER BY item_id, ts
    """)
    con.sql(f"""
        CREATE TABLE buys AS
        SELECT user_id, item_id, ts, day
        FROM '{EVENTS}'
        WHERE event = 'transaction' AND day <= DATE '{VAL_LAST_DAY}'
    """)
    con.sql("""
        CREATE TABLE buy_at_purchase AS
        SELECT b.user_id, b.day, c.category_id
        FROM buys b
        ASOF LEFT JOIN item_cat c ON b.item_id = c.item_id AND b.ts >= c.ts
        WHERE c.category_id IS NOT NULL
    """)
    con.sql(f"""
        CREATE TABLE rows AS
        SELECT e.user_id, e.item_id, e.day,
               epoch_ms(CAST(e.day AS TIMESTAMP)) + {DAY_SHIFT_MS} AS day_start,
               c.{USER_CATEGORY_BUY_RECENCY} AS got
        FROM '{EXAMPLES}' e
        JOIN '{CANDIDATES}' c USING (user_id, item_id, day)
        WHERE e.split IN ('train', 'val')
    """)
    con.sql("""
        CREATE TABLE row_strict AS
        SELECT r.*, c.category_id
        FROM rows r
        ASOF LEFT JOIN item_cat c
          ON r.item_id = c.item_id AND r.day_start > c.ts
    """)
    con.sql("""
        CREATE TABLE row_inclusive AS
        SELECT r.user_id, r.item_id, r.day, c.category_id
        FROM rows r
        ASOF LEFT JOIN item_cat c
          ON r.item_id = c.item_id AND r.day_start >= c.ts
    """)
    stats = con.sql("""
        WITH expected AS (
            SELECT r.user_id, r.item_id, r.day, r.category_id, r.got,
                   max(b.day) AS last_day
            FROM row_strict r
            LEFT JOIN buy_at_purchase b
              ON r.category_id IS NOT NULL
             AND b.user_id = r.user_id
             AND b.category_id = r.category_id
             AND b.day < r.day
            GROUP BY r.user_id, r.item_id, r.day, r.category_id, r.got
        ),
        scored AS (
            SELECT e.*, i.category_id AS inclusive_category,
                   CASE WHEN e.category_id IS NULL OR e.last_day IS NULL THEN NULL
                        ELSE date_diff('day', e.last_day, e.day)::DOUBLE
                   END AS exp
            FROM expected e
            JOIN row_inclusive i USING (user_id, item_id, day)
        ),
        same_day AS (
            SELECT DISTINCT user_id, category_id, day FROM buy_at_purchase
        )
        SELECT
            (SELECT count(*) FROM scored WHERE got IS DISTINCT FROM exp),
            (SELECT count(*) FROM scored WHERE got IS NOT NULL AND got < 1),
            (SELECT count(*) FROM scored s
             JOIN same_day d
               ON d.user_id = s.user_id AND d.category_id = s.category_id AND d.day = s.day),
            (SELECT count(*) FROM scored
             WHERE category_id IS DISTINCT FROM inclusive_category),
            (SELECT count(*) FROM scored WHERE category_id IS NULL AND got IS NOT NULL),
            (SELECT count(*) FROM scored WHERE category_id IS NOT NULL AND last_day IS NULL),
            (SELECT count(*) FROM scored WHERE got IS NOT NULL)
    """).fetchone()
    n_bad, n_zero, n_same_day, n_strict, n_unknown, n_never, n_known = stats
    assert n_bad == 0
    assert n_zero == 0
    assert n_same_day > 0
    assert n_strict > 0
    assert n_unknown == 0
    assert n_never > 0
    assert n_known > 0


@pytest.mark.needs_data(
    "data/item_properties_part1.csv",
    "data/item_properties_part2.csv",
    "data/examples.parquet",
    "data/feature_candidates.parquet",
)
def test_availability_does_not_use_same_day_snapshot():
    """available и число смен 0↔1 смотрят только снимки строго до 03:00 дня группы."""
    con = duckdb.connect()
    con.sql("""
        CREATE TABLE avail AS
        SELECT itemid AS item_id, timestamp AS ts, TRY_CAST(value AS INTEGER) AS available
        FROM read_csv_auto(
            ['data/item_properties_part1.csv', 'data/item_properties_part2.csv'],
            types={'value': 'VARCHAR', 'property': 'VARCHAR'}
        )
        WHERE property = 'available' AND TRY_CAST(value AS INTEGER) IN (0, 1)
        ORDER BY item_id, ts
    """)
    con.sql("""
        CREATE TABLE avail_cum AS
        SELECT item_id, ts, available,
               sum(CASE WHEN prev IS NULL OR prev = available THEN 0 ELSE 1 END)
                   OVER (PARTITION BY item_id ORDER BY ts) AS switches
        FROM (
            SELECT item_id, ts, available,
                   lag(available) OVER (PARTITION BY item_id ORDER BY ts) AS prev
            FROM avail
        )
        ORDER BY item_id, ts
    """)
    con.sql(f"""
        CREATE TABLE rows AS
        SELECT e.user_id, e.item_id, e.day,
               epoch_ms(CAST(e.day AS TIMESTAMP)) + {DAY_SHIFT_MS} AS day_start,
               c.{ITEM_AVAILABLE} AS got_av,
               c.{ITEM_AVAILABLE_SWITCHES} AS got_sw
        FROM '{EXAMPLES}' e
        JOIN '{CANDIDATES}' c USING (user_id, item_id, day)
        WHERE e.split IN ('train', 'val')
    """)
    stats = con.sql("""
        WITH strict AS (
            SELECT r.*, a.available AS exp_av, a.switches AS exp_sw
            FROM rows r
            ASOF LEFT JOIN avail_cum a
              ON r.item_id = a.item_id AND r.day_start > a.ts
        ),
        inclusive AS (
            SELECT r.user_id, r.item_id, r.day, a.available AS inc_av, a.switches AS inc_sw
            FROM rows r
            ASOF LEFT JOIN avail_cum a
              ON r.item_id = a.item_id AND r.day_start >= a.ts
        )
        SELECT
            (SELECT count(*) FROM strict
             WHERE got_av IS DISTINCT FROM exp_av OR got_sw IS DISTINCT FROM exp_sw),
            (SELECT count(*) FROM strict s
             JOIN inclusive i USING (user_id, item_id, day)
             WHERE s.exp_av IS DISTINCT FROM i.inc_av OR s.exp_sw IS DISTINCT FROM i.inc_sw),
            (SELECT count(*) FROM strict WHERE exp_av IS NULL),
            (SELECT count(*) FROM strict WHERE exp_sw > 0),
            (SELECT count(*) FROM strict s
             JOIN inclusive i USING (user_id, item_id, day)
             WHERE i.inc_sw = s.exp_sw + 1)
    """).fetchone()
    n_bad, n_same_day, n_missing, n_switched, n_same_day_flip = stats
    assert n_bad == 0
    assert n_same_day > 0
    assert n_missing > 0
    assert n_switched > 0
    assert n_same_day_flip > 0
