"""Второй период по предрегистрации в STATUS.md.

Списки признаков заморожены. Test не читается. Метрика модели — среднее
предсказаний пяти сидов, колонки по алфавиту.
"""

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from train_lgbm import (
    BOOTSTRAP_SEED,
    CANDIDATES_PATH,
    EARLY_STOPPING_ROUNDS,
    EXAMPLES_PATH,
    N_BOOTSTRAP,
    PARAMS,
    ROOT,
    TIE_SEED,
    bootstrap_indices,
    diff_ci,
    per_group_metrics,
)

EVENTS_PATH = ROOT / "data" / "events_clean.parquet"
MANUAL_PATH = ROOT / "data" / "features_manual.parquet"
SELECTION_PATH = ROOT / "report" / "feature_selection.csv"
OUT_PATH = ROOT / "report" / "second_period.csv"
SEEDS = (42, 43, 44, 45, 46)
# Ничья бейзлайнов — seed 0, как в baselines.py. У моделей — TIE_SEED.
BASELINE_TIE_SEED = 0

# Заморожено в предрегистрации. Порядок обучения — алфавитный, не этот.
A_NEW = (
    "user_item__addtocart__1__count",
    "user_item__addtocart__7__count",
    "user_item__addtocart__30__count",
    "user_item__view__1__count",
    "user_item__view__7__count",
    "item__addtocart__14__count",
    "item__addtocart__30__count",
    "item__addtocart__7__count",
    "item__addtocart__1__count",
    "item__view__14__count",
    "item__view__1__count",
    "item__transaction__14__count",
    "item__transaction__30__count",
    "item__view__all__recency",
    "user_item__addtocart__7_30__trend",
    "item__transaction__7__count",
    "item__available__asof",
    "item__addtocart__all__recency",
    "user_item__addtocart__all__recency",
    "item__transaction__1__share",
    "user_item__view__7_30__trend",
)
STEP4 = (
    "user_category__transaction__30__count",
    "user_category__transaction__30__share",
    "category_all__transaction__30__count",
    "category_all__transaction__repeat_share",
    "user_category__transaction__all__recency",
    "item__available__asof",
    "item__available__switches",
)
A_NEW_PLUS = A_NEW + tuple(name for name in STEP4 if name not in A_NEW)

TRAIN2_FIRST = "2015-05-03"
TRAIN2_LAST = "2015-08-06"
VAL2_FIRST = "2015-08-07"
VAL2_LAST = "2015-08-20"
VAL1_FIRST = "2015-08-21"
VAL1_LAST = "2015-09-03"


def frozen_lists_match_selection():
    table = pd.read_csv(SELECTION_PATH)
    passed = set(table.loc[table["прошёл"] == 1, "признак"])
    if passed != set(A_NEW):
        raise RuntimeError("A_new не совпал с пометкой «прошёл»")
    if len(A_NEW_PLUS) != 27:
        raise RuntimeError(f"в A_new+7 ожидалось 27 признаков, есть {len(A_NEW_PLUS)}")
    if set(STEP4) - set(A_NEW) != set(A_NEW_PLUS) - set(A_NEW):
        raise RuntimeError("добавка к A_new не совпала с семью признаками шага 4")


def load_frame(extra_features=()):
    """Train и первый val. Строки test в запрос не входят."""
    columns = sorted(set(A_NEW_PLUS) | set(extra_features))
    quoted = ", ".join(f'c."{name}"::FLOAT AS "{name}"' for name in columns)
    con = duckdb.connect()
    frame = con.sql(f"""
        SELECT e.user_id, e.item_id, e.day, e.label, e.split, {quoted}
        FROM '{EXAMPLES_PATH}' e
        JOIN '{CANDIDATES_PATH}' c USING (user_id, item_id, day)
        WHERE e.split IN ('train', 'val')
        ORDER BY e.split, e.user_id, e.day, e.item_id
    """).df()
    if frame["split"].isin(["test"]).any():
        raise RuntimeError("в выборку попал test")
    days = pd.to_datetime(frame["day"])
    if days.max() > pd.Timestamp(VAL1_LAST) or days.min() < pd.Timestamp(TRAIN2_FIRST):
        raise RuntimeError("в выборке дни вне train и первого val")
    return frame


def slice_period(frame, train_last, val_first, val_last, val_split):
    days = pd.to_datetime(frame["day"])
    train = frame[(frame["split"] == "train") & (days <= pd.Timestamp(train_last))].reset_index(drop=True)
    if val_split == "train":
        val = frame[
            (frame["split"] == "train")
            & (days >= pd.Timestamp(val_first))
            & (days <= pd.Timestamp(val_last))
        ].reset_index(drop=True)
    else:
        val = frame[frame["split"] == "val"].reset_index(drop=True)
    if len(train) == 0 or len(val) == 0:
        raise RuntimeError("пустое окно периода")
    if pd.to_datetime(train["day"]).max() >= pd.to_datetime(val["day"]).min():
        raise RuntimeError("train пересекается с val этого периода")
    return train, val


def fit_seed(train, val, features, seed, n_rounds=None):
    """Ранняя остановка на val, либо ровно n_rounds деревьев без меток val."""
    params = dict(PARAMS)
    params["random_state"] = seed
    if n_rounds is not None:
        params["n_estimators"] = int(n_rounds)
        model = lgb.LGBMClassifier(**params)
        model.fit(train[features], train["label"].to_numpy())
        scores = model.predict_proba(val[features], num_iteration=int(n_rounds))[:, 1]
        return scores, int(n_rounds)
    model = lgb.LGBMClassifier(**params)
    model.fit(
        train[features],
        train["label"].to_numpy(),
        eval_X=val[features],
        eval_y=val["label"].to_numpy(),
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False),
            lgb.log_evaluation(0),
        ],
    )
    scores = model.predict_proba(val[features], num_iteration=model.best_iteration_)[:, 1]
    return scores, int(model.best_iteration_ or 0)


def average_scores(train, val, features, n_rounds=None):
    """Среднее вероятностей пяти сидов и NDCG@3 каждого сида."""
    columns = sorted(features)
    total = None
    seed_ndcg = []
    iterations = []
    for seed in SEEDS:
        print(f"  сид {seed}", flush=True)
        scores, iteration = fit_seed(train, val, columns, seed, n_rounds=n_rounds)
        grouped = per_group_metrics(val, scores, seed=TIE_SEED)
        seed_ndcg.append(float(grouped["ndcg3"].mean()))
        iterations.append(iteration)
        total = scores if total is None else total + scores
    return total / len(SEEDS), seed_ndcg, iterations


def baseline_scores(val, event_last_day, manual_split):
    keys = val[["user_id", "item_id", "day"]].copy()
    keys["row_id"] = np.arange(len(keys))
    con = duckdb.connect()
    con.register("keys", keys)
    scored = con.sql(f"""
        WITH daily AS (
            SELECT item_id, day, count(*) AS n
            FROM '{EVENTS_PATH}'
            WHERE event IN ('addtocart', 'transaction')
              AND day <= DATE '{event_last_day}'
            GROUP BY item_id, day
        )
        SELECT k.row_id,
               coalesce(sum(d.n), 0) AS popularity,
               coalesce(any_value(f.ui_views_30d), 0) AS prior_views
        FROM keys k
        LEFT JOIN daily d
          ON d.item_id = k.item_id
         AND d.day < CAST(k.day AS DATE)
         AND d.day >= CAST(k.day AS DATE) - INTERVAL 7 DAY
        LEFT JOIN '{MANUAL_PATH}' f
          ON f.user_id = k.user_id
         AND f.item_id = k.item_id
         AND f.day = CAST(k.day AS DATE)
         AND f.split = '{manual_split}'
        GROUP BY k.row_id
        ORDER BY k.row_id
    """).df()
    if len(scored) != len(val):
        raise RuntimeError("бейзлайны не совпали со строками")
    order = val.sort_values(["user_id", "item_id", "day"]).index.to_numpy()
    random_scores = np.zeros(len(val), dtype=np.float64)
    random_scores[order] = np.random.default_rng(0).random(len(val))
    return (
        scored["popularity"].to_numpy(dtype=np.float64),
        scored["prior_views"].to_numpy(dtype=np.float64),
        random_scores,
    )


def metric_row(period, name, val, scores, tie_seed, n_features, iterations):
    grouped = per_group_metrics(val, scores, seed=tie_seed)
    return {
        "record": "metrics",
        "period": period,
        "name": name,
        "n_features": n_features,
        "n_groups": len(grouped["ndcg3"]),
        "ndcg_3": float(grouped["ndcg3"].mean()),
        "ndcg_5": float(grouped["ndcg5"].mean()),
        "mrr": float(grouped["mrr"].mean()),
        "auc": float(roc_auc_score(val["label"], scores)),
        "iterations": ",".join(str(item) for item in iterations) if iterations else "",
    }, grouped


def history_note():
    """Строки train2 и val2, у которых до дня группы меньше 30 суток от первого события.

    Первое событие — min(day) в events_clean с day не позже конца train2.
    Val1 и test в этот запрос не входят. Раньше счётчик val2 требовал день
    одновременно в августе и раньше 2015-06-02, поэтому был тождественно 0.
    """
    con = duckdb.connect()
    train_short, val_short, first_day = con.sql(f"""
        WITH bounds AS (
            SELECT min(day) AS first_day
            FROM '{EVENTS_PATH}'
            WHERE day <= DATE '{TRAIN2_LAST}'
        )
        SELECT
            (SELECT count(*)
             FROM '{EXAMPLES_PATH}' e, bounds b
             WHERE e.split = 'train'
               AND e.day BETWEEN DATE '{TRAIN2_FIRST}' AND DATE '{TRAIN2_LAST}'
               AND e.day < b.first_day + INTERVAL 30 DAY),
            (SELECT count(*)
             FROM '{EXAMPLES_PATH}' e, bounds b
             WHERE e.split = 'train'
               AND e.day BETWEEN DATE '{VAL2_FIRST}' AND DATE '{VAL2_LAST}'
               AND e.day < b.first_day + INTERVAL 30 DAY),
            (SELECT first_day FROM bounds)
    """).fetchone()
    return int(train_short), int(val_short), first_day


def evaluate_period(frame, period, train_last, val_first, val_last, val_split, event_last, manual_split):
    train, val = slice_period(frame, train_last, val_first, val_last, val_split)
    print(f"{period}: train {len(train)}  val {len(val)}", flush=True)
    popularity, prior_views, random_scores = baseline_scores(val, event_last, manual_split)
    fitted = {}
    seed_rows = []
    metric_rows = []
    for name, features in (("A_new", A_NEW), ("A_new_plus", A_NEW_PLUS)):
        print(f"обучаю {period} {name}: {len(features)} признаков", flush=True)
        scores, seed_ndcg, iterations = average_scores(train, val, features)
        row, grouped = metric_row(period, name, val, scores, TIE_SEED, len(features), iterations)
        metric_rows.append(row)
        fitted[name] = grouped
        seed_rows.append({
            "record": "seeds",
            "period": period,
            "name": name,
            "ndcg_3_min": min(seed_ndcg),
            "ndcg_3_max": max(seed_ndcg),
            "ndcg_3_seeds": ",".join(f"{value:.6f}" for value in seed_ndcg),
        })
        print(
            f"  среднее NDCG@3 {row['ndcg_3']:.6f}  сиды {seed_rows[-1]['ndcg_3_min']:.6f}–{seed_rows[-1]['ndcg_3_max']:.6f}",
            flush=True,
        )
    for name, scores in (
        ("popularity", popularity),
        ("prior_views", prior_views),
        ("random", random_scores),
    ):
        row, grouped = metric_row(period, name, val, scores, BASELINE_TIE_SEED, None, [])
        metric_rows.append(row)
        fitted[name] = grouped
    n_groups = len(fitted["A_new"]["ndcg3"])
    for name, grouped in fitted.items():
        if len(grouped["ndcg3"]) != n_groups:
            raise RuntimeError(f"{period} {name}: другое число групп")
        if not np.array_equal(grouped["user_id"], fitted["A_new"]["user_id"]):
            raise RuntimeError(f"{period} {name}: другие группы")
    indices = bootstrap_indices(n_groups, N_BOOTSTRAP, BOOTSTRAP_SEED)
    pairs = (
        ("A_new_plus", "A_new", "A_new_plus-A_new"),
        ("A_new", "popularity", "A_new-popularity"),
        ("A_new_plus", "popularity", "A_new_plus-popularity"),
    )
    diff_rows = []
    for left, right, name in pairs:
        estimate, low, high = diff_ci(fitted[left]["ndcg3"], fitted[right]["ndcg3"], indices)
        diff_rows.append({
            "record": "diff",
            "period": period,
            "name": name,
            "n_groups": n_groups,
            "ndcg_3": estimate,
            "ndcg_3_low": low,
            "ndcg_3_high": high,
        })
    return metric_rows, diff_rows, seed_rows


def main():
    frozen_lists_match_selection()
    frame = load_frame()
    rows = []
    # Сначала второй период: по нему правило выбирает набор.
    metrics, diffs, seeds = evaluate_period(
        frame, "val2", TRAIN2_LAST, VAL2_FIRST, VAL2_LAST, "train", VAL2_LAST, "train"
    )
    rows.extend(metrics + diffs + seeds)
    choice = "A_new_plus" if diffs[0]["ndcg_3_low"] > 0 else "A_new"
    rows.append({
        "record": "decision",
        "period": "val2",
        "name": choice,
        "ndcg_3": diffs[0]["ndcg_3"],
        "ndcg_3_low": diffs[0]["ndcg_3_low"],
        "ndcg_3_high": diffs[0]["ndcg_3_high"],
    })
    metrics, diffs, seeds = evaluate_period(
        frame, "val1", "2015-08-20", VAL1_FIRST, VAL1_LAST, "val", VAL1_LAST, "val"
    )
    rows.extend(metrics + diffs + seeds)
    train_short, val_short, _first_day = history_note()
    rows.append({
        "record": "history",
        "period": "train2",
        "name": "rows_before_30d",
        "n_groups": train_short,
    })
    rows.append({
        "record": "history",
        "period": "val2",
        "name": "rows_before_30d",
        "n_groups": val_short,
    })
    report = pd.DataFrame(rows)
    report.to_csv(OUT_PATH, index=False)
    print(f"выбран: {choice}")
    print(f"строк train2 с историей короче 30 дней: {train_short}; на val2 таких строк: {val_short}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
