import duckdb

# No event before day D. 999 is past the 138-day span of the log, so "never"
# stays distinguishable from a real gap and is not stored as NULL.
RECENCY_MISSING = 999

con = duckdb.connect()
con.sql("CREATE TABLE ex AS SELECT * FROM 'data/examples.parquet'")


def prefix(name, path, keys, metrics):
    """Cumulative sums ordered by day.

    The total on [D-w, D) is the prefix at the last day before D minus the
    prefix at the last day before D-w. ASOF finds those two days, so we never
    join every past day to every example.
    """
    key_sql = ", ".join(keys)
    metric_sql = ", ".join(metrics)
    cum_sql = ", ".join(
        f"sum({m}) OVER (PARTITION BY {key_sql} ORDER BY day) AS {m}_cum"
        for m in metrics
    )
    con.sql(f"""
        CREATE TABLE {name} AS
        SELECT {key_sql}, day, {metric_sql}, {cum_sql}
        FROM '{path}'
        ORDER BY {key_sql}, day
    """)


def span(hi, lo, metric):
    return f"coalesce({hi}.{metric}_cum, 0) - coalesce({lo}.{metric}_cum, 0)"


def asof_before(alias, table, keys, day_sql):
    match = " AND ".join(f"ex.{k} = {alias}.{k}" for k in keys)
    return f"ASOF LEFT JOIN {table} {alias} ON {match} AND {day_sql} > {alias}.day"


def cutoff(days):
    return f"CAST(ex.day - INTERVAL {days} DAY AS DATE)"


prefix("user_cum", "data/daily_user.parquet", ["user_id"], ["views", "carts", "buys"])
prefix("item_cum", "data/daily_item.parquet", ["item_id"], ["views", "carts", "buys"])
prefix("ui_cum", "data/daily_user_item.parquet", ["user_id", "item_id"], ["views", "carts", "buys"])
prefix("uc_cum", "data/daily_user_cat.parquet", ["user_id", "category_id"], ["views", "buys"])
prefix("cat_cum", "data/daily_cat.parquet", ["category_id"], ["views"])
con.sql("""
    CREATE TABLE ui_view_days AS
    SELECT user_id, item_id, day
    FROM 'data/daily_user_item.parquet'
    WHERE views > 0
    ORDER BY user_id, item_id, day
""")
print("prefix tables ready")

con.sql(f"""
    CREATE TABLE feat_user AS
    SELECT ex.user_id, ex.item_id, ex.day,
           {span('u_hi', 'u_7', 'views')} AS user_views_7d,
           {span('u_hi', 'u_30', 'views')} AS user_views_30d,
           {span('u_hi', 'u_7', 'carts')} AS user_carts_7d,
           {span('u_hi', 'u_30', 'carts')} AS user_carts_30d,
           {span('u_hi', 'u_7', 'buys')} AS user_buys_7d,
           {span('u_hi', 'u_30', 'buys')} AS user_buys_30d,
           CASE WHEN {span('u_hi', 'u_30', 'views')} = 0 THEN 0.0
                ELSE ({span('u_hi', 'u_30', 'carts')} + {span('u_hi', 'u_30', 'buys')})::DOUBLE
                     / {span('u_hi', 'u_30', 'views')}
           END AS user_conv_30d,
           coalesce(date_diff('day', u_hi.day, ex.day), {RECENCY_MISSING})
               AS user_days_since_activity
    FROM ex
    {asof_before('u_hi', 'user_cum', ['user_id'], 'ex.day')}
    {asof_before('u_7', 'user_cum', ['user_id'], cutoff(7))}
    {asof_before('u_30', 'user_cum', ['user_id'], cutoff(30))}
""")
print("feat_user:", con.sql("SELECT count(*) FROM feat_user").fetchone()[0])

con.sql(f"""
    CREATE TABLE feat_item AS
    SELECT ex.user_id, ex.item_id, ex.day,
           {span('i_hi', 'i_1', 'views')} AS item_views_1d,
           {span('i_hi', 'i_7', 'views')} AS item_views_7d,
           {span('i_hi', 'i_30', 'views')} AS item_views_30d,
           {span('i_hi', 'i_1', 'carts')} AS item_carts_1d,
           {span('i_hi', 'i_7', 'carts')} AS item_carts_7d,
           {span('i_hi', 'i_30', 'carts')} AS item_carts_30d,
           {span('i_hi', 'i_1', 'buys')} AS item_buys_1d,
           {span('i_hi', 'i_7', 'buys')} AS item_buys_7d,
           {span('i_hi', 'i_30', 'buys')} AS item_buys_30d,
           CASE WHEN {span('i_hi', 'i_30', 'views')} = 0 THEN 0.0
                ELSE ({span('i_hi', 'i_30', 'carts')} + {span('i_hi', 'i_30', 'buys')})::DOUBLE
                     / {span('i_hi', 'i_30', 'views')}
           END AS item_conv_30d
    FROM ex
    {asof_before('i_hi', 'item_cum', ['item_id'], 'ex.day')}
    {asof_before('i_1', 'item_cum', ['item_id'], cutoff(1))}
    {asof_before('i_7', 'item_cum', ['item_id'], cutoff(7))}
    {asof_before('i_30', 'item_cum', ['item_id'], cutoff(30))}
""")
print("feat_item:", con.sql("SELECT count(*) FROM feat_item").fetchone()[0])

con.sql(f"""
    CREATE TABLE feat_ui AS
    SELECT ex.user_id, ex.item_id, ex.day,
           {span('p_hi', 'p_1', 'views')} AS ui_views_1d,
           {span('p_hi', 'p_7', 'views')} AS ui_views_7d,
           {span('p_hi', 'p_30', 'views')} AS ui_views_30d,
           coalesce(date_diff('day', seen.day, ex.day), {RECENCY_MISSING})
               AS ui_days_since_view,
           (coalesce(p_hi.carts_cum, 0) > 0)::INTEGER AS ui_cart_before
    FROM ex
    {asof_before('p_hi', 'ui_cum', ['user_id', 'item_id'], 'ex.day')}
    {asof_before('p_1', 'ui_cum', ['user_id', 'item_id'], cutoff(1))}
    {asof_before('p_7', 'ui_cum', ['user_id', 'item_id'], cutoff(7))}
    {asof_before('p_30', 'ui_cum', ['user_id', 'item_id'], cutoff(30))}
    {asof_before('seen', 'ui_view_days', ['user_id', 'item_id'], 'ex.day')}
""")
print("feat_ui:", con.sql("SELECT count(*) FROM feat_ui").fetchone()[0])

# Numbers are filled for every row. Rows whose category was not known before
# day D are turned back to NULL in the final SELECT: a fallback category comes
# from a property record written after that day.
con.sql(f"""
    CREATE TABLE feat_uc AS
    SELECT ex.user_id, ex.item_id, ex.day,
           {span('c_hi', 'c_30', 'views')} AS uc_views_30d,
           {span('c_hi', 'c_30', 'buys')} AS uc_buys_30d
    FROM ex
    {asof_before('c_hi', 'uc_cum', ['user_id', 'category_id'], 'ex.day')}
    {asof_before('c_30', 'uc_cum', ['user_id', 'category_id'], cutoff(30))}
""")
con.sql(f"""
    CREATE TABLE feat_cat AS
    SELECT ex.user_id, ex.item_id, ex.day,
           {span('g_hi', 'g_7', 'views')} AS cat_views_7d
    FROM ex
    {asof_before('g_hi', 'cat_cum', ['category_id'], 'ex.day')}
    {asof_before('g_7', 'cat_cum', ['category_id'], cutoff(7))}
""")
print("feat_uc:", con.sql("SELECT count(*) FROM feat_uc").fetchone()[0])

con.sql("""
    COPY (
        SELECT ex.user_id, ex.item_id, ex.day, ex.split, ex.label, ex.rank_group,
               u.user_views_7d, u.user_views_30d,
               u.user_carts_7d, u.user_carts_30d,
               u.user_buys_7d, u.user_buys_30d,
               u.user_conv_30d, u.user_days_since_activity,
               i.item_views_1d, i.item_views_7d, i.item_views_30d,
               i.item_carts_1d, i.item_carts_7d, i.item_carts_30d,
               i.item_buys_1d, i.item_buys_7d, i.item_buys_30d,
               i.item_conv_30d,
               ui.ui_views_1d, ui.ui_views_7d, ui.ui_views_30d,
               ui.ui_days_since_view, ui.ui_cart_before,
               CASE WHEN ex.category_fallback OR ex.category_id IS NULL THEN NULL
                    ELSE uc.uc_views_30d END AS uc_views_30d,
               CASE WHEN ex.category_fallback OR ex.category_id IS NULL THEN NULL
                    ELSE uc.uc_buys_30d END AS uc_buys_30d,
               CASE WHEN ex.category_fallback OR ex.category_id IS NULL THEN NULL
                    WHEN u.user_views_30d = 0 THEN 0.0
                    ELSE uc.uc_views_30d::DOUBLE / u.user_views_30d
               END AS uc_view_share_30d,
               CASE WHEN ex.category_fallback OR ex.category_id IS NULL THEN NULL
                    ELSE c.cat_views_7d END AS cat_views_7d
        FROM ex
        JOIN feat_user u USING (user_id, item_id, day)
        JOIN feat_item i USING (user_id, item_id, day)
        JOIN feat_ui ui USING (user_id, item_id, day)
        JOIN feat_uc uc USING (user_id, item_id, day)
        JOIN feat_cat c USING (user_id, item_id, day)
    ) TO 'data/features_manual.parquet' (FORMAT parquet)
""")

n = con.sql("SELECT count(*) FROM 'data/features_manual.parquet'").fetchone()[0]
ncols = len(con.sql("DESCRIBE SELECT * FROM 'data/features_manual.parquet'").fetchall())
print(f"features_manual: {n} rows, {ncols} columns")
con.sql("""
    SELECT round(avg(uc_views_30d IS NULL::INT), 4) AS uc_null_share
    FROM 'data/features_manual.parquet'
""").show()
