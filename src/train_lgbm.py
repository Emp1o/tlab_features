from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from metrics import ndcg_at_k, reciprocal_rank

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PATH = ROOT / "data" / "examples.parquet"
CANDIDATES_PATH = ROOT / "data" / "feature_candidates.parquet"
SELECTION_PATH = ROOT / "report" / "feature_selection.csv"
REPORT_PATH = ROOT / "report" / "lgbm_comparison.csv"

# Границы сплита те же, что в prepare.py. Test сюда не входит.
TRAIN_LAST = pd.Timestamp("2015-08-20")
VAL_FIRST = pd.Timestamp("2015-08-21")
VAL_LAST = pd.Timestamp("2015-09-03")
KEYS = ("user_id", "item_id", "day", "label", "split")

# Один набор для A, B и C: иначе разница метрик может быть из-за настроек, а не из-за признаков.
PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "random_state": 42,
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 200,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "verbosity": -1,
}
EARLY_STOPPING_ROUNDS = 50
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 42
TIE_SEED = 42


def load_keys():
    """Строки train и val. Test в запрос не входит."""
    con = duckdb.connect()
    return con.sql(f"""
        SELECT user_id, item_id, day, label, split
        FROM '{EXAMPLES_PATH}'
        WHERE split IN ('train', 'val')
    """).df()


def ensure_train_rows(df):
    if len(df) == 0 or not df["split"].eq("train").all():
        raise RuntimeError("в обучении есть строки не из train")
    if pd.to_datetime(df["day"]).max() > TRAIN_LAST:
        raise RuntimeError("в обучении есть дни после train")


def ensure_val_rows(df):
    if len(df) == 0 or not df["split"].eq("val").all():
        raise RuntimeError("оценка использует не только val")
    days = pd.to_datetime(df["day"])
    if days.min() < VAL_FIRST or days.max() > VAL_LAST:
        raise RuntimeError("в val попали дни вне 21.08–03.09")


def candidate_names():
    described = duckdb.sql(f"DESCRIBE SELECT * FROM '{CANDIDATES_PATH}'").fetchall()
    return [row[0] for row in described if row[0] not in ("user_id", "item_id", "day")]


def selected_names():
    table = pd.read_csv(SELECTION_PATH)
    return table.loc[table["прошёл"] == 1, "признак"].tolist()


def feature_sets(all_features, selected):
    missing = [name for name in selected if name not in set(all_features)]
    if missing:
        raise RuntimeError(f"в кандидатах нет отобранных признаков: {missing[:5]}")
    # Порядок A — порядок силы в csv, а не порядок столбцов.
    chosen = [name for name in selected]
    user_features = [name for name in all_features if name.startswith("user__")]
    with_user = chosen + [name for name in user_features if name not in set(chosen)]
    return {"A": chosen, "B": with_user, "C": list(all_features)}


def load_modeling_frame(feature_names):
    keys = load_keys()
    con = duckdb.connect()
    con.register("keys", keys)
    quoted = ", ".join(f'c."{name}"::FLOAT AS "{name}"' for name in feature_names)
    frame = con.sql(f"""
        SELECT k.user_id, k.item_id, k.day, k.label, k.split, {quoted}
        FROM keys k
        JOIN '{CANDIDATES_PATH}' c USING (user_id, item_id, day)
        ORDER BY k.split, k.user_id, k.day, k.item_id
    """).df()
    if len(frame) != len(keys):
        raise RuntimeError("число строк признаков не совпало с train+val")
    return frame


def fit_model(train, val, features):
    ensure_train_rows(train)
    ensure_val_rows(val)
    model = lgb.LGBMClassifier(**PARAMS)
    # Остановка по logloss на val. Test в eval_set не передаём.
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
    return model, scores


def per_group_metrics(df, scores, seed=TIE_SEED):
    """NDCG и MRR по группам из двух и более товаров с хотя бы одной покупкой.

    Группа из одного товара в ранжирование не входит: сравнивать нечего.
    Группа без покупки в среднее не входит и не считается нулём.
    """
    rng = np.random.default_rng(seed)
    work = df[["user_id", "item_id", "day", "label"]].copy()
    work["score"] = scores
    ndcg3, ndcg5, mrrs, sizes = [], [], [], []
    for _, group in work.groupby(["user_id", "day"], sort=True):
        if len(group) < 2 or int(group["label"].sum()) < 1:
            continue
        group = group.sort_values("item_id")
        noise = rng.random(len(group))
        order = np.lexsort((noise, -group["score"].to_numpy()))
        labels = group["label"].to_numpy()[order]
        ndcg3.append(ndcg_at_k(labels, 3))
        ndcg5.append(ndcg_at_k(labels, 5))
        mrrs.append(reciprocal_rank(labels))
        sizes.append(len(group))
    return {
        "ndcg3": np.asarray(ndcg3, dtype=np.float64),
        "ndcg5": np.asarray(ndcg5, dtype=np.float64),
        "mrr": np.asarray(mrrs, dtype=np.float64),
        "size": np.asarray(sizes, dtype=np.int32),
    }


def size_bucket_means(ndcg3, size):
    return {
        "2": float(ndcg3[size == 2].mean()) if np.any(size == 2) else np.nan,
        "3-5": float(ndcg3[(size >= 3) & (size <= 5)].mean()) if np.any((size >= 3) & (size <= 5)) else np.nan,
        ">5": float(ndcg3[size > 5].mean()) if np.any(size > 5) else np.nan,
    }


def bootstrap_indices(n_groups, n_reps, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_groups, size=(n_reps, n_groups))


def mean_ci(values, indices):
    draws = values[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def diff_ci(left, right, indices):
    draws = left[indices].mean(axis=1) - right[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(left.mean() - right.mean()), float(low), float(high)


def top_importance(model, features, k=15):
    gain = model.booster_.feature_importance(importance_type="gain")
    table = pd.DataFrame({"feature": features, "gain": gain})
    table = table.sort_values(["gain", "feature"], ascending=[False, True], ignore_index=True)
    return table.head(k)


def main():
    all_features = candidate_names()
    sets = feature_sets(all_features, selected_names())
    frame = load_modeling_frame(all_features)
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    val = frame[frame["split"] == "val"].reset_index(drop=True)
    ensure_train_rows(train)
    ensure_val_rows(val)
    print(f"train: {len(train)}  val: {len(val)}  признаков всего: {len(all_features)}")

    fitted = {}
    group_ndcg = {}
    metric_rows = []
    importance_rows = []
    for name in ("A", "B", "C"):
        features = sets[name]
        print(f"обучаю {name}: {len(features)} признаков", flush=True)
        model, scores = fit_model(train, val, features)
        grouped = per_group_metrics(val, scores)
        buckets = size_bucket_means(grouped["ndcg3"], grouped["size"])
        auc = float(roc_auc_score(val["label"], scores))
        fitted[name] = {
            "n_features": len(features),
            "best_iteration": int(model.best_iteration_ or PARAMS["n_estimators"]),
            "ndcg5": float(grouped["ndcg5"].mean()),
            "mrr": float(grouped["mrr"].mean()),
            "auc": auc,
            "buckets": buckets,
            "n_groups": int(len(grouped["ndcg3"])),
        }
        group_ndcg[name] = grouped["ndcg3"]
        top = top_importance(model, features)
        for rank, row in top.iterrows():
            importance_rows.append({
                "record": "importance",
                "set": name,
                "rank": int(rank) + 1,
                "feature": row["feature"],
                "gain": float(row["gain"]),
            })
        print(f"  лучшая итерация {fitted[name]['best_iteration']}, групп {fitted[name]['n_groups']}")

    if len({len(group_ndcg[name]) for name in ("A", "B", "C")}) != 1:
        raise RuntimeError("наборы посчитаны по разным группам")
    # Одни и те же группы в каждом повторе, иначе разница B−A смешивает состав выборок.
    indices = bootstrap_indices(len(group_ndcg["A"]), N_BOOTSTRAP, BOOTSTRAP_SEED)
    summaries = {}
    for name in ("A", "B", "C"):
        estimate, low, high = mean_ci(group_ndcg[name], indices)
        summaries[name] = (estimate, low, high)
        info = fitted[name]
        metric_rows.append({
            "record": "metrics",
            "set": name,
            "n_features": info["n_features"],
            "best_iteration": info["best_iteration"],
            "n_groups": info["n_groups"],
            "ndcg_3": estimate,
            "ndcg_3_low": low,
            "ndcg_3_high": high,
            "ndcg_5": info["ndcg5"],
            "mrr": info["mrr"],
            "auc": info["auc"],
            "ndcg_3_size_2": info["buckets"]["2"],
            "ndcg_3_size_3_5": info["buckets"]["3-5"],
            "ndcg_3_size_gt5": info["buckets"][">5"],
        })

    for left, right in (("B", "A"), ("C", "A"), ("C", "B")):
        estimate, low, high = diff_ci(group_ndcg[left], group_ndcg[right], indices)
        summaries[f"{left}-{right}"] = (estimate, low, high)
        metric_rows.append({
            "record": "metrics",
            "set": f"{left}-{right}",
            "n_features": np.nan,
            "best_iteration": np.nan,
            "n_groups": len(group_ndcg["A"]),
            "ndcg_3": estimate,
            "ndcg_3_low": low,
            "ndcg_3_high": high,
            "ndcg_5": np.nan,
            "mrr": np.nan,
            "auc": np.nan,
            "ndcg_3_size_2": np.nan,
            "ndcg_3_size_3_5": np.nan,
            "ndcg_3_size_gt5": np.nan,
        })

    metrics = pd.DataFrame(metric_rows)
    importance = pd.DataFrame(importance_rows)
    report = pd.concat([metrics, importance], ignore_index=True)
    report.to_csv(REPORT_PATH, index=False)

    show = metrics.copy()
    for column in (
        "ndcg_3", "ndcg_3_low", "ndcg_3_high", "ndcg_5", "mrr", "auc",
        "ndcg_3_size_2", "ndcg_3_size_3_5", "ndcg_3_size_gt5",
    ):
        show[column] = show[column].round(4)
    print(show.to_string(index=False))
    for name in ("A", "B", "C"):
        print(f"топ-15 по gain, набор {name}:")
        part = importance[importance["set"] == name][["rank", "feature", "gain"]]
        print(part.to_string(index=False))


if __name__ == "__main__":
    main()
