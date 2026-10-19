"""NDCG@3 моделей против бейзлайнов на одних и тех же группах val.

Test в запросы не входит. Бутстреп тот же, что у разниц B−A, C−A и C−B:
1000 повторов, seed 42, одни и те же индексы групп.
Ничья у моделей разрывается seed 42, у бейзлайнов — seed 0, как в их таблице.
"""

import duckdb
import numpy as np
import pandas as pd

from train_lgbm import (
    BOOTSTRAP_SEED,
    N_BOOTSTRAP,
    ROOT,
    bootstrap_indices,
    candidate_names,
    diff_ci,
    ensure_train_rows,
    ensure_val_rows,
    feature_sets,
    fit_model,
    load_modeling_frame,
    per_group_metrics,
    selected_names,
)

EVENTS_PATH = ROOT / "data" / "events_clean.parquet"
MANUAL_PATH = ROOT / "data" / "features_manual.parquet"
OUT_PATH = ROOT / "report" / "baseline_diffs.csv"
# Последний день val. События test (с 2015-09-04) в суммы не входят.
VAL_LAST_DAY = "2015-09-03"

PAIRS = (
    ("A", "popularity", "A-popularity"),
    ("B", "popularity", "B-popularity"),
    ("C", "popularity", "C-popularity"),
    ("A", "prior_views", "A-prior_views"),
)


def baseline_scores(val):
    """Популярность и прошлые просмотры в порядке строк val."""
    keys = val[["user_id", "item_id", "day"]].copy()
    keys["row_id"] = np.arange(len(keys))
    con = duckdb.connect()
    con.register("val_keys", keys)
    scored = con.sql(f"""
        WITH daily AS (
            SELECT item_id, day, count(*) AS item_carts_buys_7d
            FROM '{EVENTS_PATH}'
            WHERE event IN ('addtocart', 'transaction')
              AND day <= DATE '{VAL_LAST_DAY}'
            GROUP BY item_id, day
        )
        SELECT k.row_id,
               coalesce(sum(d.item_carts_buys_7d), 0) AS popularity,
               coalesce(any_value(f.ui_views_30d), 0) AS prior_views
        FROM val_keys k
        LEFT JOIN daily d
          ON d.item_id = k.item_id
         AND d.day < CAST(k.day AS DATE)
         AND d.day >= CAST(k.day AS DATE) - INTERVAL 7 DAY
        LEFT JOIN '{MANUAL_PATH}' f
          ON f.user_id = k.user_id
         AND f.item_id = k.item_id
         AND f.day = CAST(k.day AS DATE)
         AND f.split = 'val'
        GROUP BY k.row_id
        ORDER BY k.row_id
    """).df()
    if len(scored) != len(val) or not np.array_equal(scored["row_id"].to_numpy(), keys["row_id"].to_numpy()):
        raise RuntimeError("оценки бейзлайнов не совпали со строками val")
    if scored[["popularity", "prior_views"]].isna().any().any():
        raise RuntimeError("в оценках бейзлайнов есть пропуски")
    return scored["popularity"].to_numpy(dtype=np.float64), scored["prior_views"].to_numpy(dtype=np.float64)


def same_groups(grouped):
    """Все оценки стоят на одних и тех же группах в одном порядке."""
    names = list(grouped)
    ref = grouped[names[0]]
    n = len(ref["ndcg3"])
    for name in names[1:]:
        other = grouped[name]
        if len(other["ndcg3"]) != n:
            raise RuntimeError(f"у {name} другое число групп")
        if not np.array_equal(other["user_id"], ref["user_id"]) or not np.array_equal(other["day"], ref["day"]):
            raise RuntimeError(f"группы {name} не совпали с {names[0]}")
    return n


def comparison_rows(ndcg, indices):
    rows = []
    for left, right, name in PAIRS:
        estimate, low, high = diff_ci(ndcg[left], ndcg[right], indices)
        rows.append({
            "comparison": name,
            "n_groups": len(ndcg[left]),
            "ndcg_3": estimate,
            "ndcg_3_low": low,
            "ndcg_3_high": high,
        })
    return pd.DataFrame(rows)


def main():
    all_features = candidate_names()
    sets = feature_sets(all_features, selected_names())
    frame = load_modeling_frame(all_features)
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    val = frame[frame["split"] == "val"].reset_index(drop=True)
    ensure_train_rows(train)
    ensure_val_rows(val)

    popularity, prior_views = baseline_scores(val)
    grouped = {}
    for name in ("A", "B", "C"):
        print(f"обучаю {name}: {len(sets[name])} признаков", flush=True)
        _, scores = fit_model(train, val, sets[name])
        grouped[name] = per_group_metrics(val, scores)
        print(f"  NDCG@3 {grouped[name]['ndcg3'].mean():.6f}", flush=True)
    # Seed 0 — тот же жребий, что в evaluate() и в таблице бейзлайнов.
    # У A/B/C жребий seed 42, как в train_lgbm: оценка разницы равна разности этих NDCG.
    grouped["popularity"] = per_group_metrics(val, popularity, seed=0)
    grouped["prior_views"] = per_group_metrics(val, prior_views, seed=0)
    n_groups = same_groups(grouped)
    print(f"групп: {n_groups}")
    print(f"popularity NDCG@3 {grouped['popularity']['ndcg3'].mean():.6f}")
    print(f"prior_views NDCG@3 {grouped['prior_views']['ndcg3'].mean():.6f}")

    indices = bootstrap_indices(n_groups, N_BOOTSTRAP, BOOTSTRAP_SEED)
    table = comparison_rows({name: grouped[name]["ndcg3"] for name in grouped}, indices)
    table.to_csv(OUT_PATH, index=False)
    show = table.copy()
    for column in ("ndcg_3", "ndcg_3_low", "ndcg_3_high"):
        show[column] = show[column].round(4)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
