import duckdb

# Та же граница суток, что в prepare.py и daily.py: день начинается в 03:00 UTC.
DAY_SHIFT_MS = 3 * 3600 * 1000

EVENTS = (
    ("view", "views"),
    ("addtocart", "carts"),
    ("transaction", "buys"),
)
WINDOWS = (1, 7, 14, 30)
CANDIDATES_PATH = "data/feature_candidates.parquet"

# nunique — число разных товаров, поэтому только у пользователя и категории.
# У давности нет окна. Тренд — одно отношение count_7 / count_30.
NUNIQUE_WHOS = {"user", "category"}
# Покупки пользователя в категории товара за 30 дней до дня группы.
USER_CATEGORY_BUY_COUNT = "user_category__transaction__30__count"
USER_CATEGORY_BUY_SHARE = "user_category__transaction__30__share"
# Те же покупки, но по всем пользователям категории.
CATEGORY_ALL_BUY_COUNT = "category_all__transaction__30__count"
CATEGORY_ALL_REPEAT_SHARE = "category_all__transaction__repeat_share"
# Дни с последней покупки пользователя в категории товара. Не покупал — NULL.
USER_CATEGORY_BUY_RECENCY = "user_category__transaction__all__recency"
# Меньше 5 покупок в истории категории: (повторные + 1) / (покупки + 2).
REPEAT_RAW_MIN = 5


def prefix(con, name, path, keys):
    """Накопленные числа событий по дням.

    Сумма на [D-w, D) — префикс последнего дня до D минус префикс последнего
    дня до D-w. Эти два дня находит ASOF.
    """
    key_sql = ", ".join(keys)
    cum = ", ".join(
        f"sum({metric}) OVER (PARTITION BY {key_sql} ORDER BY day) AS {metric}_cum"
        for _, metric in EVENTS
    )
    raw = ", ".join(metric for _, metric in EVENTS)
    con.sql(f"""
        CREATE TABLE {name} AS
        SELECT {key_sql}, day, {raw}, {cum}
        FROM '{path}'
        ORDER BY {key_sql}, day
    """)


def span(hi, lo, metric):
    return f"coalesce({hi}.{metric}_cum, 0) - coalesce({lo}.{metric}_cum, 0)"


def asof_before(alias, table, keys, day_sql):
    match = " AND ".join(f"ex.{key} = {alias}.{key}" for key in keys)
    return f"ASOF LEFT JOIN {table} {alias} ON {match} AND {day_sql} > {alias}.day"


def cutoff(days):
    return f"CAST(ex.day - INTERVAL {days} DAY AS DATE)"


def event_days(con, name, path, keys):
    """Строка есть только в те дни, когда это событие случилось."""
    key_sql = ", ".join(keys)
    tables = {}
    for event, metric in EVENTS:
        table = f"{name}_{event}"
        con.sql(f"""
            CREATE TABLE {table} AS
            SELECT {key_sql}, day
            FROM '{path}'
            WHERE {metric} > 0
            ORDER BY {key_sql}, day
        """)
        tables[event] = table
    return tables


def value_columns(who):
    """Формулы count, share, recency и trend. Без nunique."""
    hi = "hi"
    alias = {1: "w1", 7: "w7", 14: "w14", 30: "w30"}
    columns = []
    for event, metric in EVENTS:
        counts = {w: span(hi, alias[w], metric) for w in WINDOWS}
        for w in WINDOWS:
            columns.append((f"{who}__{event}__{w}__count", counts[w]))
            # В окне не было событий: доля неизвестна, это не настоящий ноль.
            total = " + ".join(span(hi, alias[w], m) for _, m in EVENTS)
            share = (
                f"CASE WHEN ({total}) = 0 THEN NULL "
                f"ELSE ({counts[w]})::DOUBLE / ({total}) END"
            )
            columns.append((f"{who}__{event}__{w}__share", share))
        columns.append(
            (f"{who}__{event}__all__recency", f"date_diff('day', r_{event}.day, ex.day)")
        )
        trend = (
            f"CASE WHEN ({counts[30]}) = 0 THEN NULL "
            f"ELSE ({counts[7]})::DOUBLE / ({counts[30]}) END"
        )
        columns.append((f"{who}__{event}__7_30__trend", trend))
    return columns


def asof_joins(cum_table, day_tables, keys):
    joins = [asof_before("hi", cum_table, keys, "ex.day")]
    joins.extend(
        asof_before(alias, cum_table, keys, cutoff(w))
        for w, alias in ((1, "w1"), (7, "w7"), (14, "w14"), (30, "w30"))
    )
    joins.extend(
        asof_before(f"r_{event}", day_tables[event], keys, "ex.day") for event, _ in EVENTS
    )
    return "\n".join(joins)


def sql_columns(columns, mask_category):
    lines = []
    for name, expr in columns:
        if mask_category:
            # У fallback-строки category_id взят из более поздней записи свойств.
            expr = (
                "CASE WHEN ex.category_fallback OR ex.category_id IS NULL "
                f"THEN NULL ELSE ({expr}) END"
            )
        lines.append(f"({expr})::DOUBLE AS {name}")
    return ",\n".join(lines)


def nunique_columns(who):
    lines = []
    for event, metric in EVENTS:
        for w in WINDOWS:
            lines.append(
                f"""count(DISTINCT CASE
                    WHEN d.{metric} > 0
                     AND d.day >= CAST(k.day - INTERVAL {w} DAY AS DATE)
                    THEN d.item_id END)
                    AS {who}__{event}__{w}__nunique"""
            )
    return ",\n".join(lines)


def add_user_category_purchase_features(con):
    """Пять признаков покупок в категории товара строки.

    Категория строки — последнее categoryid с timestamp строго до начала дня D.
    У каждой прошлой покупки своя категория, as-of на момент покупки.
    День D и более поздние события не входят. Нет категории до дня D — NULL.
    Личная доля — от всех покупок пользователя за 30 дней; если их ноль, доля NULL.
    Доля повторных — по всей истории категории до дня D. Покупка повторная, если
    у этого покупателя уже была покупка в той же категории с меньшим timestamp.
    Если таких покупок не было, доля NULL. Если их меньше 5, к числителю и
    знаменателю добавляется по единице.
    Давность — число дней между днём группы и днём последней покупки этого
    пользователя в категории строки. Покупки дня группы не входят. Не покупал — NULL.
    """
    con.sql("""
        CREATE TABLE buy_at_ts AS
        SELECT e.user_id, e.ts, e.day, c.category_id
        FROM (
            SELECT user_id, item_id, ts, day
            FROM 'data/events_clean.parquet'
            WHERE event = 'transaction'
        ) e
        ASOF LEFT JOIN item_cat c
          ON e.item_id = c.item_id AND e.ts >= c.ts
        WHERE c.category_id IS NOT NULL
    """)
    con.sql("""
        CREATE TABLE buy_repeat AS
        SELECT user_id, category_id, day,
               count(*) OVER (
                   PARTITION BY user_id, category_id
                   ORDER BY ts
                   RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
               ) > 0 AS is_repeat
        FROM buy_at_ts
    """)
    con.sql("""
        CREATE TABLE uc_buy_cum AS
        SELECT user_id, category_id, day, buys,
               sum(buys) OVER (
                   PARTITION BY user_id, category_id ORDER BY day
               ) AS buys_cum
        FROM (
            SELECT user_id, category_id, day, count(*) AS buys
            FROM buy_at_ts
            GROUP BY user_id, category_id, day
        )
        ORDER BY user_id, category_id, day
    """)
    con.sql("""
        CREATE TABLE cat_buy_cum AS
        SELECT category_id, day,
               sum(n) OVER (PARTITION BY category_id ORDER BY day) AS n_cum,
               sum(n_repeat) OVER (PARTITION BY category_id ORDER BY day) AS repeat_cum
        FROM (
            SELECT category_id, day,
                   count(*) AS n,
                   count(*) FILTER (WHERE is_repeat) AS n_repeat
            FROM buy_repeat
            GROUP BY category_id, day
        )
        ORDER BY category_id, day
    """)
    con.sql("""
        CREATE TABLE uc_buy_day AS
        SELECT DISTINCT user_id, category_id, day
        FROM buy_at_ts
        ORDER BY user_id, category_id, day
    """)
    con.sql(f"""
        CREATE TABLE row_cat AS
        SELECT e.user_id, e.item_id, e.day, c.category_id
        FROM (
            SELECT user_id, item_id, day,
                   epoch_ms(CAST(day AS TIMESTAMP)) + {DAY_SHIFT_MS} AS day_start_ms
            FROM ex
        ) e
        ASOF LEFT JOIN item_cat c
          ON e.item_id = c.item_id AND e.day_start_ms > c.ts
    """)
    con.sql(f"""
        CREATE TABLE ex_cat AS
        SELECT ex.user_id, ex.item_id, ex.day, ex.split,
               rc.category_id AS row_category_id,
               fu.user__transaction__30__count AS user_buys_30
        FROM ex
        JOIN feat_user fu USING (user_id, item_id, day)
        JOIN row_cat rc USING (user_id, item_id, day)
    """)
    con.sql(f"""
        CREATE TABLE feat_user_category AS
        SELECT ex.user_id, ex.item_id, ex.day,
               CASE WHEN ex.row_category_id IS NULL THEN NULL
                    ELSE (coalesce(hi.buys_cum, 0) - coalesce(lo.buys_cum, 0))::DOUBLE
               END AS {USER_CATEGORY_BUY_COUNT},
               CASE WHEN ex.row_category_id IS NULL THEN NULL
                    WHEN ex.user_buys_30 = 0 THEN NULL
                    ELSE (coalesce(hi.buys_cum, 0) - coalesce(lo.buys_cum, 0))::DOUBLE
                         / ex.user_buys_30
               END AS {USER_CATEGORY_BUY_SHARE},
               CASE WHEN ex.row_category_id IS NULL THEN NULL
                    ELSE (coalesce(chi.n_cum, 0) - coalesce(clo.n_cum, 0))::DOUBLE
               END AS {CATEGORY_ALL_BUY_COUNT},
               CASE WHEN ex.row_category_id IS NULL OR coalesce(chi.n_cum, 0) = 0 THEN NULL
                    WHEN chi.n_cum < {REPEAT_RAW_MIN}
                    THEN (chi.repeat_cum + 1)::DOUBLE / (chi.n_cum + 2)
                    ELSE chi.repeat_cum::DOUBLE / chi.n_cum
               END AS {CATEGORY_ALL_REPEAT_SHARE},
               CASE WHEN ex.row_category_id IS NULL OR last_buy.day IS NULL THEN NULL
                    ELSE date_diff('day', last_buy.day, ex.day)::DOUBLE
               END AS {USER_CATEGORY_BUY_RECENCY}
        FROM ex_cat ex
        ASOF LEFT JOIN uc_buy_cum hi
          ON ex.user_id = hi.user_id
         AND ex.row_category_id = hi.category_id
         AND ex.day > hi.day
        ASOF LEFT JOIN uc_buy_cum lo
          ON ex.user_id = lo.user_id
         AND ex.row_category_id = lo.category_id
         AND {cutoff(30)} > lo.day
        ASOF LEFT JOIN cat_buy_cum chi
          ON ex.row_category_id = chi.category_id
         AND ex.day > chi.day
        ASOF LEFT JOIN cat_buy_cum clo
          ON ex.row_category_id = clo.category_id
         AND {cutoff(30)} > clo.day
        ASOF LEFT JOIN uc_buy_day last_buy
          ON ex.user_id = last_buy.user_id
         AND ex.row_category_id = last_buy.category_id
         AND ex.day > last_buy.day
    """)
    n_feat = con.sql("SELECT count(*) FROM feat_user_category").fetchone()[0]
    n_ex = con.sql("SELECT count(*) FROM ex").fetchone()[0]
    if n_feat != n_ex:
        raise SystemExit(f"user-category features row count {n_feat} != examples {n_ex}")
    print("user-category purchases:", n_feat)


def main():
    con = duckdb.connect()
    con.sql("SET preserve_insertion_order = false")
    con.sql("CREATE TABLE ex AS SELECT * FROM 'data/examples.parquet'")
    con.sql(f"""
        CREATE TABLE item_cat AS
        SELECT item_id, ts, category_id
        FROM 'data/item_category.parquet'
        WHERE category_id IS NOT NULL
        ORDER BY item_id, ts
    """)

    prefix(con, "user_cum", "data/daily_user.parquet", ["user_id"])
    prefix(con, "item_cum", "data/daily_item.parquet", ["item_id"])
    prefix(con, "ui_cum", "data/daily_user_item.parquet", ["user_id", "item_id"])
    prefix(con, "cat_cum", "data/daily_cat.parquet", ["category_id"])
    user_days = event_days(con, "user_ev", "data/daily_user.parquet", ["user_id"])
    item_days = event_days(con, "item_ev", "data/daily_item.parquet", ["item_id"])
    ui_days = event_days(
        con, "ui_ev", "data/daily_user_item.parquet", ["user_id", "item_id"]
    )
    cat_days = event_days(con, "cat_ev", "data/daily_cat.parquet", ["category_id"])
    print("prefix tables ready")

    # Число разных товаров вычитанием префиксов не получить.
    con.sql("CREATE TABLE user_day AS SELECT DISTINCT user_id, day FROM ex")
    con.sql(f"""
        CREATE TABLE user_nunique AS
        SELECT k.user_id, k.day,
               {nunique_columns("user")}
        FROM user_day k
        LEFT JOIN 'data/daily_user_item.parquet' d
          ON d.user_id = k.user_id
         AND d.day < k.day
         AND d.day >= CAST(k.day - INTERVAL 30 DAY AS DATE)
        GROUP BY k.user_id, k.day
    """)
    print("user nunique:", con.sql("SELECT count(*) FROM user_nunique").fetchone()[0])

    con.sql(f"""
        CREATE TABLE item_day_cat AS
        SELECT d.item_id, d.day, d.views, d.carts, d.buys, c.category_id
        FROM (
            SELECT *, epoch_ms(CAST(day AS TIMESTAMP)) + {DAY_SHIFT_MS} AS day_start_ms
            FROM 'data/daily_item.parquet'
        ) d
        ASOF LEFT JOIN item_cat c
          ON d.item_id = c.item_id AND d.day_start_ms >= c.ts
        WHERE c.category_id IS NOT NULL
    """)
    con.sql("""
        CREATE TABLE cat_day AS
        SELECT DISTINCT category_id, day
        FROM ex
        WHERE category_id IS NOT NULL AND NOT category_fallback
    """)
    con.sql(f"""
        CREATE TABLE cat_nunique AS
        SELECT k.category_id, k.day,
               {nunique_columns("category")}
        FROM cat_day k
        LEFT JOIN item_day_cat d
          ON d.category_id = k.category_id
         AND d.day < k.day
         AND d.day >= CAST(k.day - INTERVAL 30 DAY AS DATE)
        GROUP BY k.category_id, k.day
    """)
    print("category nunique:", con.sql("SELECT count(*) FROM cat_nunique").fetchone()[0])

    specs = (
        ("user", "user_cum", ["user_id"], user_days, "user_nunique", "nu.user_id = ex.user_id AND nu.day = ex.day", False),
        ("item", "item_cum", ["item_id"], item_days, None, None, False),
        ("user_item", "ui_cum", ["user_id", "item_id"], ui_days, None, None, False),
        ("category", "cat_cum", ["category_id"], cat_days, "cat_nunique", "nu.category_id = ex.category_id AND nu.day = ex.day", True),
    )
    for who, cum, keys, days, nunique_table, nunique_on, mask in specs:
        columns = value_columns(who)
        extra_join = ""
        if nunique_table:
            columns.extend(
                (f"{who}__{event}__{w}__nunique", f"nu.{who}__{event}__{w}__nunique")
                for event, _ in EVENTS
                for w in WINDOWS
            )
            extra_join = f"LEFT JOIN {nunique_table} nu ON {nunique_on}"
        con.sql(f"""
            CREATE TABLE feat_{who} AS
            SELECT ex.user_id, ex.item_id, ex.day,
                   {sql_columns(columns, mask)}
            FROM ex
            {asof_joins(cum, days, keys)}
            {extra_join}
        """)
        n = con.sql(f"SELECT count(*) FROM feat_{who}").fetchone()[0]
        print(f"feat_{who}: {n}")

    add_user_category_purchase_features(con)

    # Смена внутри [D-30, D) — два разных значения. Значение прямо перед окном
    # учитываем, записи с начала дня D и позже — нет.
    con.sql(f"""
        CREATE TABLE item_day AS
        SELECT DISTINCT item_id, day,
               epoch_ms(CAST(day AS TIMESTAMP)) + {DAY_SHIFT_MS} AS day_start,
               epoch_ms(CAST(day AS TIMESTAMP) - INTERVAL 30 DAY) + {DAY_SHIFT_MS} AS win_start
        FROM ex
    """)
    con.sql("""
        CREATE TABLE cat_in_window AS
        SELECT b.item_id, b.day, c.category_id
        FROM item_day b
        JOIN item_cat c
          ON c.item_id = b.item_id AND c.ts >= b.win_start AND c.ts < b.day_start
        UNION ALL
        SELECT b.item_id, b.day, prev.category_id
        FROM item_day b
        ASOF LEFT JOIN item_cat prev
          ON b.item_id = prev.item_id AND b.win_start > prev.ts
        WHERE prev.category_id IS NOT NULL
    """)
    con.sql("""
        CREATE TABLE item_changed AS
        SELECT item_id, day,
               (count(DISTINCT category_id) >= 2)::DOUBLE AS item__category__30__changed
        FROM cat_in_window
        GROUP BY item_id, day
    """)
    print("category changes:", con.sql("SELECT count(*) FROM item_changed").fetchone()[0])

    feature_sql = []
    for who, *_ in specs:
        cols = [name for name, _ in value_columns(who)]
        if who in NUNIQUE_WHOS:
            cols.extend(
                f"{who}__{event}__{w}__nunique"
                for event, _ in EVENTS
                for w in WINDOWS
            )
        feature_sql.extend(f"f_{who}.{name}" for name in cols)
        if who == "item":
            feature_sql.append(
                "coalesce(ch.item__category__30__changed, 0)::DOUBLE AS item__category__30__changed"
            )
    feature_sql.extend([
        f"f_ucat.{USER_CATEGORY_BUY_COUNT}",
        f"f_ucat.{USER_CATEGORY_BUY_SHARE}",
        f"f_ucat.{CATEGORY_ALL_BUY_COUNT}",
        f"f_ucat.{CATEGORY_ALL_REPEAT_SHARE}",
        f"f_ucat.{USER_CATEGORY_BUY_RECENCY}",
    ])

    con.sql(f"""
        COPY (
            SELECT ex.user_id, ex.item_id, ex.day,
                   {", ".join(feature_sql)}
            FROM ex
            JOIN feat_user f_user USING (user_id, item_id, day)
            JOIN feat_item f_item USING (user_id, item_id, day)
            JOIN feat_user_item f_user_item USING (user_id, item_id, day)
            JOIN feat_category f_category USING (user_id, item_id, day)
            JOIN feat_user_category f_ucat USING (user_id, item_id, day)
            LEFT JOIN item_changed ch
              ON ch.item_id = ex.item_id AND ch.day = ex.day
        ) TO '{CANDIDATES_PATH}' (FORMAT parquet)
    """)

    described = con.sql(f"DESCRIBE SELECT * FROM '{CANDIDATES_PATH}'").fetchall()
    n_rows = con.sql(f"SELECT count(*) FROM '{CANDIDATES_PATH}'").fetchone()[0]
    n_candidates = len(described) - 3
    print(f"rows: {n_rows}")
    print(f"candidates: {n_candidates}")

    bad = con.sql(f"""
        SELECT count(*) FROM '{CANDIDATES_PATH}'
        WHERE user__view__7__count > user__view__30__count
           OR item__transaction__14__count > item__transaction__30__count
           OR user__view__7__share < 0 OR user__view__7__share > 1
           OR {USER_CATEGORY_BUY_COUNT} < 0
           OR {USER_CATEGORY_BUY_SHARE} < 0
           OR {USER_CATEGORY_BUY_SHARE} > 1
           OR (user__transaction__30__count = 0 AND {USER_CATEGORY_BUY_SHARE} IS NOT NULL)
           OR ({USER_CATEGORY_BUY_COUNT} IS NULL AND {USER_CATEGORY_BUY_SHARE} IS NOT NULL)
           OR (
                {USER_CATEGORY_BUY_COUNT} IS NOT NULL
                AND user__transaction__30__count > 0
                AND abs({USER_CATEGORY_BUY_SHARE} * user__transaction__30__count
                        - {USER_CATEGORY_BUY_COUNT}) > 1e-6
           )
           OR ({USER_CATEGORY_BUY_COUNT} IS NULL) <> ({CATEGORY_ALL_BUY_COUNT} IS NULL)
           OR {CATEGORY_ALL_BUY_COUNT} < 0
           OR {CATEGORY_ALL_REPEAT_SHARE} < 0
           OR {CATEGORY_ALL_REPEAT_SHARE} > 1
           OR ({CATEGORY_ALL_BUY_COUNT} IS NULL AND {CATEGORY_ALL_REPEAT_SHARE} IS NOT NULL)
           OR {USER_CATEGORY_BUY_RECENCY} < 1
           OR ({CATEGORY_ALL_BUY_COUNT} IS NULL AND {USER_CATEGORY_BUY_RECENCY} IS NOT NULL)
           OR (
                {USER_CATEGORY_BUY_COUNT} > 0
                AND ({USER_CATEGORY_BUY_RECENCY} IS NULL OR {USER_CATEGORY_BUY_RECENCY} > 30)
           )
    """).fetchone()[0]
    if bad:
        raise SystemExit(f"feature sanity check failed on {bad} rows")


if __name__ == "__main__":
    main()
