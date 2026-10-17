import sys

import duckdb
import numpy as np
import pandas as pd

from metrics import evaluate

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"
if SPLIT not in ("train", "val"):
    sys.exit(
        f"Отказ: сплит '{SPLIT}' оценивать нельзя. Допустимы только train и val."
    )

con = duckdb.connect()
df = con.sql(f"""
    WITH ex AS (SELECT * FROM 'data/examples.parquet' WHERE split = '{SPLIT}'),
    daily AS (
        SELECT item_id, day, count(*) AS item_carts_buys_7d
        FROM 'data/events_clean.parquet'
        WHERE event IN ('addtocart', 'transaction')
        GROUP BY item_id, day
    )
    SELECT ex.user_id, ex.item_id, ex.day, ex.label, ex.rank_group,
           coalesce(sum(d.item_carts_buys_7d), 0) AS item_carts_buys_7d,
           coalesce(any_value(f.ui_views_30d), 0) AS ui_views_30d
    FROM ex
    LEFT JOIN daily d
      ON d.item_id = ex.item_id AND d.day < ex.day AND d.day >= ex.day - INTERVAL 7 DAY
    LEFT JOIN 'data/features_manual.parquet' f
      ON f.user_id = ex.user_id AND f.item_id = ex.item_id AND f.day = ex.day
    GROUP BY ex.user_id, ex.item_id, ex.day, ex.label, ex.rank_group
    ORDER BY ex.user_id, ex.item_id, ex.day
""").df()

# One draw per row position, so the frame must arrive in a fixed order.
df["random"] = np.random.default_rng(0).random(len(df))

rows = {
    "random": evaluate(df, "random"),
    "popularity (carts+buys 7d)": evaluate(df, "item_carts_buys_7d"),
    "prior views (ui_views_30d)": evaluate(df, "ui_views_30d"),
}
print(f"Split: {SPLIT}, rows: {len(df)}")
print(pd.DataFrame(rows).T.round(4).to_string())
