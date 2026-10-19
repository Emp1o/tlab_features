import duckdb

DAY_SHIFT_MS = 3 * 3600 * 1000  # сутки с 03:00 до 03:00 UTC
BOT_MIN_EVENTS = 200
VAL_DAYS, TEST_DAYS = 14, 14

con = duckdb.connect()

con.sql(f"""
    CREATE TABLE raw AS
    SELECT visitorid AS user_id, itemid AS item_id, event, timestamp AS ts,
           CAST(epoch_ms(timestamp - {DAY_SHIFT_MS}) AS DATE) AS day
    FROM read_csv_auto('data/events.csv')
""")

# Границы сплита берутся из всего лога, до фильтра ботов. Правило бота смотрит
# только на train: корзина на val или test — будущая метка, по ней нельзя решать,
# кого выкидывать из всех частей.
last_day = con.sql("SELECT max(day) FROM raw").fetchone()[0]
train_last = con.sql(
    f"SELECT DATE '{last_day}' - INTERVAL {VAL_DAYS + TEST_DAYS} DAY"
).fetchone()[0]

con.sql(f"""
    CREATE TABLE bots_all AS
    SELECT user_id FROM raw GROUP BY user_id
    HAVING count(*) >= {BOT_MIN_EVENTS}
       AND count(*) FILTER (WHERE event IN ('addtocart', 'transaction')) = 0
""")
print("Bots before (full period):", con.sql("SELECT count(*) FROM bots_all").fetchone()[0],
      "| their events:",
      con.sql("SELECT count(*) FROM raw WHERE user_id IN (SELECT user_id FROM bots_all)").fetchone()[0])

con.sql(f"""
    CREATE TABLE bots AS
    SELECT user_id FROM raw
    WHERE day <= DATE '{train_last}'
    GROUP BY user_id
    HAVING count(*) >= {BOT_MIN_EVENTS}
       AND count(*) FILTER (WHERE event IN ('addtocart', 'transaction')) = 0
""")
print("Train last day:", train_last)
print("Bots after (train only):", con.sql("SELECT count(*) FROM bots").fetchone()[0],
      "| their events:",
      con.sql("SELECT count(*) FROM raw WHERE user_id IN (SELECT user_id FROM bots)").fetchone()[0])

con.sql("CREATE TABLE ev AS SELECT * FROM raw WHERE user_id NOT IN (SELECT user_id FROM bots)")
con.sql("COPY ev TO 'data/events_clean.parquet' (FORMAT parquet)")

# Журнал категории: значение и момент, когда оно стало известно
con.sql("""
    CREATE TABLE item_cat AS
    SELECT itemid AS item_id, timestamp AS ts, TRY_CAST(value AS INTEGER) AS category_id
    FROM read_csv_auto(['data/item_properties_part1.csv', 'data/item_properties_part2.csv'],
                       types={'value': 'VARCHAR', 'property': 'VARCHAR'})
    WHERE property = 'categoryid'
""")
con.sql("COPY item_cat TO 'data/item_category.parquet' (FORMAT parquet)")
con.sql("COPY (SELECT * FROM read_csv_auto('data/category_tree.csv')) TO 'data/category_tree.parquet' (FORMAT parquet)")

con.sql("""
    CREATE TABLE examples AS
    SELECT user_id, item_id, day,
           max(CASE WHEN event IN ('addtocart', 'transaction') THEN 1 ELSE 0 END) AS label
    FROM ev GROUP BY user_id, item_id, day
""")

# Категория, известная до начала дня примера; самое раннее значение — запасное
con.sql(f"""
    CREATE TABLE examples_cat AS
    SELECT e.*, c.category_id AS cat_asof
    FROM (SELECT *, epoch_ms(CAST(day AS TIMESTAMP)) + {DAY_SHIFT_MS} AS day_start_ms FROM examples) e
    ASOF LEFT JOIN item_cat c ON e.item_id = c.item_id AND e.day_start_ms >= c.ts
""")
con.sql("""
    CREATE TABLE first_cat AS
    SELECT item_id, arg_min(category_id, ts) AS category_id FROM item_cat GROUP BY item_id
""")

con.sql(f"""
    CREATE TABLE dataset AS
    SELECT e.user_id, e.item_id, e.day, e.label,
           coalesce(e.cat_asof, f.category_id) AS category_id,
           e.cat_asof IS NULL AND f.category_id IS NOT NULL AS category_fallback,
           CASE WHEN e.day > DATE '{last_day}' - INTERVAL {TEST_DAYS} DAY THEN 'test'
                WHEN e.day > DATE '{last_day}' - INTERVAL {TEST_DAYS + VAL_DAYS} DAY THEN 'val'
                ELSE 'train' END AS split
    FROM examples_cat e LEFT JOIN first_cat f USING (item_id)
""")

# Группы для метрик ранжирования: хотя бы 2 товара и хотя бы один положительный
con.sql("""
    CREATE TABLE dataset_final AS
    SELECT d.*, (g.n_items >= 2 AND g.n_pos >= 1) AS rank_group
    FROM dataset d JOIN (
        SELECT user_id, day, count(*) AS n_items, sum(label) AS n_pos
        FROM dataset GROUP BY user_id, day
    ) g USING (user_id, day)
""")
con.sql("COPY dataset_final TO 'data/examples.parquet' (FORMAT parquet)")

con.sql("""
    SELECT split, min(day) AS first_day, max(day) AS last_day,
           count(*) AS rows, round(avg(label), 4) AS pos_share,
           count(DISTINCT (user_id, day)) FILTER (WHERE rank_group) AS rank_groups
    FROM dataset_final GROUP BY split ORDER BY first_day
""").show()
con.sql("""
    SELECT round(avg(category_id IS NULL::INT), 4) AS no_category,
           round(avg(category_fallback::INT), 4) AS category_fallback
    FROM dataset_final
""").show()