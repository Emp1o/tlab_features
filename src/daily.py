import duckdb

DAY_SHIFT_MS = 3 * 3600 * 1000  # та же граница суток, что в prepare.py: 03:00 UTC

con = duckdb.connect()

# Одна строка на сущность и день. Признаки суммируют эти строки по окну,
# а не сканируют сырой лог для каждого примера.
COUNTS = """
    count(*) FILTER (WHERE event = 'view') AS views,
    count(*) FILTER (WHERE event = 'addtocart') AS carts,
    count(*) FILTER (WHERE event = 'transaction') AS buys
"""

con.sql(f"""
    CREATE TABLE ev AS
    SELECT user_id, item_id, event, day FROM 'data/events_clean.parquet'
""")

for name, keys in (
    ("daily_user", "user_id, day"),
    ("daily_item", "item_id, day"),
    ("daily_user_item", "user_id, item_id, day"),
):
    con.sql(f"""
        COPY (
            SELECT {keys}, {COUNTS}
            FROM ev GROUP BY {keys}
        ) TO 'data/{name}.parquet' (FORMAT parquet)
    """)
    n = con.sql(f"SELECT count(*) FROM 'data/{name}.parquet'").fetchone()[0]
    print(f"{name}: {n} rows")

# Категория, известная к началу дня события (тот же ASOF, что в prepare.py).
# События без такой категории выбрасываем: иначе признак мог бы взять категорию из будущего.
con.sql("""
    CREATE TABLE item_cat AS
    SELECT item_id, ts, category_id
    FROM 'data/item_category.parquet'
    WHERE category_id IS NOT NULL
    ORDER BY item_id, ts
""")
con.sql(f"""
    CREATE TABLE ev_cat AS
    SELECT e.user_id, e.event, e.day, c.category_id
    FROM (
        SELECT *, epoch_ms(CAST(day AS TIMESTAMP)) + {DAY_SHIFT_MS} AS day_start_ms
        FROM ev
    ) e
    ASOF LEFT JOIN item_cat c ON e.item_id = c.item_id AND e.day_start_ms >= c.ts
    WHERE c.category_id IS NOT NULL
""")

con.sql(f"""
    COPY (
        SELECT user_id, category_id, day, {COUNTS}
        FROM ev_cat GROUP BY user_id, category_id, day
    ) TO 'data/daily_user_cat.parquet' (FORMAT parquet)
""")
con.sql(f"""
    COPY (
        SELECT category_id, day, {COUNTS}
        FROM ev_cat GROUP BY category_id, day
    ) TO 'data/daily_cat.parquet' (FORMAT parquet)
""")
for name in ("daily_user_cat", "daily_cat"):
    n = con.sql(f"SELECT count(*) FROM 'data/{name}.parquet'").fetchone()[0]
    print(f"{name}: {n} rows")
