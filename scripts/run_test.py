"""Один прогон на test. Не запускать, пока Максим не напишет предрегистрацию.

В STATUS.md стоит строка: «Предрегистрация теста (гипотеза и текст вывода на оба исхода): ждёт Максима».

Обучение — все строки split='train' с day <= 2015-08-20. Признаки — замороженный
A_NEW, колонки по алфавиту. Деревьев ровно FIXED_ROUNDS: это медиана
best_iteration A_new на val1 по сидам 42–46. Метки test и val в fit не входят.
Остальные параметры — PARAMS из src/train_lgbm.py. Сиды 42–46, метрика модели —
среднее вероятностей. Оценка — split='test': NDCG@3, NDCG@5, MRR, AUC и
бутстреп A_new − popularity (1000 повторов, seed 42).
"""

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from second_period import A_NEW, SEEDS, metric_row  # noqa: E402
from train_lgbm import (  # noqa: E402
    BOOTSTRAP_SEED,
    CANDIDATES_PATH,
    EXAMPLES_PATH,
    N_BOOTSTRAP,
    PARAMS,
    TIE_SEED,
    bootstrap_indices,
    diff_ci,
    per_group_metrics,
)
import lightgbm as lgb  # noqa: E402

EVENTS_PATH = ROOT / "data" / "events_clean.parquet"
OUT_PATH = ROOT / "report" / "test_run.csv"
TRAIN_LAST = "2015-08-20"
# Медиана best_iteration A_new на val1 по сидам 42–46: 77, 80, 91, 84, 78.
FIXED_ROUNDS = 80


def load_split(split_sql):
    columns = sorted(A_NEW)
    quoted = ", ".join(f'c."{name}"::FLOAT AS "{name}"' for name in columns)
    con = duckdb.connect()
    frame = con.sql(f"""
        SELECT e.user_id, e.item_id, e.day, e.label, e.split, {quoted}
        FROM '{EXAMPLES_PATH}' e
        JOIN '{CANDIDATES_PATH}' c USING (user_id, item_id, day)
        WHERE {split_sql}
        ORDER BY e.user_id, e.day, e.item_id
    """).df()
    return frame


def popularity_scores(val):
    keys = val[["user_id", "item_id", "day"]].copy()
    keys["row_id"] = np.arange(len(keys))
    con = duckdb.connect()
    con.register("keys", keys)
    scored = con.sql(f"""
        WITH daily AS (
            SELECT item_id, day, count(*) AS n
            FROM '{EVENTS_PATH}'
            WHERE event IN ('addtocart', 'transaction')
            GROUP BY item_id, day
        )
        SELECT k.row_id, coalesce(sum(d.n), 0) AS popularity
        FROM keys k
        LEFT JOIN daily d
          ON d.item_id = k.item_id
         AND d.day < CAST(k.day AS DATE)
         AND d.day >= CAST(k.day AS DATE) - INTERVAL 7 DAY
        GROUP BY k.row_id
        ORDER BY k.row_id
    """).df()
    if len(scored) != len(val):
        raise RuntimeError("popularity не совпала со строками test")
    return scored["popularity"].to_numpy(dtype=np.float64)


def fit_scores(train, test, features):
    if FIXED_ROUNDS is None:
        raise RuntimeError("FIXED_ROUNDS не задан")
    columns = sorted(features)
    total = None
    seed_ndcg = []
    for seed in SEEDS:
        print(f"сид {seed}", flush=True)
        params = dict(PARAMS)
        params["random_state"] = seed
        params["n_estimators"] = int(FIXED_ROUNDS)
        model = lgb.LGBMClassifier(**params)
        model.fit(train[columns], train["label"].to_numpy())
        scores = model.predict_proba(test[columns], num_iteration=int(FIXED_ROUNDS))[:, 1]
        grouped = per_group_metrics(test, scores, seed=TIE_SEED)
        seed_ndcg.append(float(grouped["ndcg3"].mean()))
        total = scores if total is None else total + scores
    return total / len(SEEDS), seed_ndcg


def main():
    train = load_split(f"e.split = 'train' AND e.day <= DATE '{TRAIN_LAST}'")
    test = load_split("e.split = 'test'")
    if len(train) == 0 or len(test) == 0:
        raise RuntimeError("пустое обучение или test")
    if not train["split"].eq("train").all() or pd.to_datetime(train["day"]).max() > pd.Timestamp(TRAIN_LAST):
        raise RuntimeError("в обучении есть дни после 2015-08-20")
    if not test["split"].eq("test").all() or pd.to_datetime(test["day"]).min() <= pd.Timestamp(TRAIN_LAST):
        raise RuntimeError("в test попали дни обучения")
    print(f"train {len(train)}  test {len(test)}  деревьев {FIXED_ROUNDS}", flush=True)
    scores, seed_ndcg = fit_scores(train, test, A_NEW)
    row, grouped = metric_row("test", "A_new", test, scores, TIE_SEED, len(A_NEW), [FIXED_ROUNDS] * len(SEEDS))
    popularity = popularity_scores(test)
    pop_row, pop_grouped = metric_row("test", "popularity", test, popularity, 0, None, [])
    if len(grouped["ndcg3"]) != len(pop_grouped["ndcg3"]):
        raise RuntimeError("разное число групп")
    if not np.array_equal(grouped["user_id"], pop_grouped["user_id"]):
        raise RuntimeError("разные группы")
    indices = bootstrap_indices(len(grouped["ndcg3"]), N_BOOTSTRAP, BOOTSTRAP_SEED)
    estimate, low, high = diff_ci(grouped["ndcg3"], pop_grouped["ndcg3"], indices)
    rows = [
        row,
        pop_row,
        {
            "record": "diff",
            "period": "test",
            "name": "A_new-popularity",
            "n_groups": len(grouped["ndcg3"]),
            "ndcg_3": estimate,
            "ndcg_3_low": low,
            "ndcg_3_high": high,
        },
        {
            "record": "seeds",
            "period": "test",
            "name": "A_new",
            "ndcg_3_min": min(seed_ndcg),
            "ndcg_3_max": max(seed_ndcg),
            "ndcg_3_seeds": ",".join(f"{value:.6f}" for value in seed_ndcg),
            "iterations": ",".join(str(FIXED_ROUNDS) for _ in SEEDS),
        },
    ]
    for item in rows:
        item.setdefault("record", "metrics")
    report = pd.DataFrame(rows)
    report.to_csv(OUT_PATH, index=False)
    print(report.to_string(index=False))
    print(f"A_new − popularity: {estimate:.6f} ({low:.6f}–{high:.6f})")
    print(f"записано {OUT_PATH}")


if __name__ == "__main__":
    main()
