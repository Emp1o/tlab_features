"""Числа лога для раздела «Почему RetailRocket».

Читает data/events.csv и пишет report/dataset_sparsity.csv.
Покупатели и медиана — по сырому логу. Группы ранжирования — как в prepare.py:
сутки с 03:00 UTC, боты только по train (≥200 событий и ни одной корзины
или покупки до конца train), группа (user_id, day) с ≥2 товарами и ≥1 покупкой.
"""

import csv
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT / "data" / "events.csv"
OUT = ROOT / "report" / "dataset_sparsity.csv"
DAY_SHIFT_MS = 3 * 3600 * 1000


def main():
    con = duckdb.connect()
    con.sql(f"""
        CREATE TABLE raw AS
        SELECT visitorid AS user_id, itemid AS item_id, event,
               CAST(epoch_ms(timestamp - {DAY_SHIFT_MS}) AS DATE) AS day
        FROM read_csv_auto('{EVENTS}')
    """)
    events, users, transactions, n_types = con.sql("""
        SELECT count(*),
               count(DISTINCT user_id),
               count(*) FILTER (WHERE event = 'transaction'),
               count(DISTINCT event)
        FROM raw
    """).fetchone()
    median_events = con.sql("""
        SELECT quantile_cont(n, 0.5)
        FROM (SELECT count(*) AS n FROM raw GROUP BY user_id)
    """).fetchone()[0]
    buyers, buyers_le2 = con.sql("""
        WITH u AS (
            SELECT count(DISTINCT item_id) FILTER (WHERE event = 'transaction') AS items,
                   count(*) FILTER (WHERE event = 'transaction') AS buys
            FROM raw GROUP BY user_id
        )
        SELECT count(*) FILTER (WHERE buys > 0),
               count(*) FILTER (WHERE buys > 0 AND items <= 2)
        FROM u
    """).fetchone()

    con.sql("""
        CREATE TABLE bots AS
        SELECT user_id FROM raw
        WHERE day <= DATE '2015-08-20'
        GROUP BY user_id
        HAVING count(*) >= 200
           AND count(*) FILTER (WHERE event IN ('addtocart', 'transaction')) = 0
    """)
    con.sql("""
        CREATE TABLE examples AS
        SELECT user_id, item_id, day,
               max(CASE WHEN event = 'transaction' THEN 1 ELSE 0 END) AS label
        FROM raw
        WHERE user_id NOT IN (SELECT user_id FROM bots)
        GROUP BY user_id, item_id, day
    """)
    con.sql("""
        CREATE TABLE grouped AS
        SELECT user_id, day,
               CASE WHEN day > DATE '2015-09-03' THEN 'test'
                    WHEN day > DATE '2015-08-20' THEN 'val'
                    ELSE 'train' END AS split,
               (count(*) >= 2 AND sum(label) >= 1) AS rank_group
        FROM examples
        GROUP BY user_id, day
    """)
    rank = dict(con.sql("""
        SELECT split, count(*) FILTER (WHERE rank_group)
        FROM grouped
        GROUP BY split
    """).fetchall())
    # val2 — кусок train, не отдельный сплит prepare.py.
    rank_val2 = con.sql("""
        SELECT count(*) FILTER (WHERE rank_group)
        FROM grouped
        WHERE day BETWEEN DATE '2015-08-07' AND DATE '2015-08-20'
    """).fetchone()[0]

    rows = [
        ("events", events, "сырой лог, все типы"),
        ("users", users, "уникальные visitorid"),
        ("transaction_events", transactions, "события transaction"),
        ("transaction_share", f"{transactions / events:.6f}", "transaction_events / events"),
        ("event_types", n_types, "view, addtocart, transaction"),
        ("median_events_per_user", median_events, "квантиль 0.5 числа событий на пользователя"),
        ("buyers", buyers, "пользователи с хотя бы одной покупкой"),
        ("buyers_at_most_2_items", buyers_le2, "из покупателей: 1 или 2 разных товара"),
        ("rank_groups_train", rank["train"], "2015-05-03 — 2015-08-20, включая окно val2"),
        ("rank_groups_val", rank["val"], "2015-08-21 — 2015-09-03"),
        ("rank_groups_val2", rank_val2, "2015-08-07 — 2015-08-20"),
        ("rank_groups_test", rank["test"], "2015-09-04 — 2015-09-17"),
    ]
    with OUT.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "note"])
        writer.writerows(rows)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
