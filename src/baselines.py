import sys

import duckdb
import numpy as np
import pandas as pd

from metrics import evaluate

SPLIT = sys.argv[1] if len(sys.argv) > 1 else "val"

con = duckdb.connect()
df = con.sql(f"""
    WITH ex AS (SELECT * FROM 'data/examples.parquet' WHERE split = '{SPLIT}'),
    daily AS (
        SELECT item_id, day, count(*) AS buys
        FROM 'data/events_clean.parquet'
        WHERE event IN ('addtocart', 'transaction')
        GROUP BY item_id, day
    )
    SELECT ex.user_id, ex.item_id, ex.day, ex.label, ex.rank_group,
           coalesce(sum(d.buys), 0) AS item_buys_7d
    FROM ex LEFT JOIN daily d
      ON d.item_id = ex.item_id AND d.day < ex.day AND d.day >= ex.day - INTERVAL 7 DAY
    GROUP BY ex.user_id, ex.item_id, ex.day, ex.label, ex.rank_group
""").df()

df["random"] = np.random.default_rng(0).random(len(df))

rows = {"random": evaluate(df, "random"),
        "popularity (7d)": evaluate(df, "item_buys_7d")}
print(f"Split: {SPLIT}, rows: {len(df)}")
print(pd.DataFrame(rows).T.round(4).to_string())