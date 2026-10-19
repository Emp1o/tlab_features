"""A_new и A_old с двумя признаками категории против A_old и popularity.

Test в обучение и в метрики не входит. Бутстреп — те же 1000 индексов, seed 42.
Ничья у моделей — seed 42, у popularity — seed 0, как в таблице бейзлайнов.
"""

import duckdb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from baseline_diffs import baseline_scores
from feature_generator import (
    CATEGORY_ALL_BUY_COUNT,
    CATEGORY_ALL_REPEAT_SHARE,
    USER_CATEGORY_BUY_COUNT,
    USER_CATEGORY_BUY_SHARE,
)
from train_lgbm import (
    BOOTSTRAP_SEED,
    CANDIDATES_PATH,
    EXAMPLES_PATH,
    N_BOOTSTRAP,
    ROOT,
    TIE_SEED,
    bootstrap_indices,
    candidate_names,
    diff_ci,
    ensure_train_rows,
    ensure_val_rows,
    fit_model,
    load_modeling_frame,
    per_group_metrics,
    selected_names,
)

OUT_PATH = ROOT / "report" / "category_comparison.csv"
SELECTION_DIFF_PATH = ROOT / "report" / "category_selection_diff.csv"
SELECTION_PATH = ROOT / "report" / "feature_selection.csv"
PERSONAL_FEATURES = (USER_CATEGORY_BUY_COUNT, USER_CATEGORY_BUY_SHARE)
CATEGORY_FEATURES = (CATEGORY_ALL_BUY_COUNT, CATEGORY_ALL_REPEAT_SHARE)
NEW_FEATURES = PERSONAL_FEATURES + CATEGORY_FEATURES

# Набор A до шага 4, порядок силы из feature_selection.csv того прогона.
A_OLD = (
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
    "item__addtocart__all__recency",
    "user_item__addtocart__all__recency",
    "item__transaction__1__share",
    "user_item__view__7_30__trend",
)

PAIRS = (
    ("A_new", "A_old", "A_new-A_old"),
    ("A_old_cat", "A_old", "A_old_cat-A_old"),
    ("A_old_personal", "A_old", "A_old_personal-A_old"),
    ("A_old_category", "A_old", "A_old_category-A_old"),
    ("A_new", "popularity", "A_new-popularity"),
    ("A_old_cat", "popularity", "A_old_cat-popularity"),
    ("C_new", "A_new", "C_new-A_new"),
)


def model_sets(all_features, selected):
    known = set(all_features)
    a_old = list(A_OLD)
    a_new = list(selected)
    a_old_personal = a_old + list(PERSONAL_FEATURES)
    a_old_category = a_old + list(CATEGORY_FEATURES)
    a_old_cat = a_old + list(NEW_FEATURES)
    missing = [name for name in a_old_cat + a_new if name not in known]
    if missing:
        raise RuntimeError(f"в кандидатах нет признаков: {missing}")
    return {
        "A_old": a_old,
        "A_new": a_new,
        "A_old_personal": a_old_personal,
        "A_old_category": a_old_category,
        "A_old_cat": a_old_cat,
        "C_new": list(all_features),
    }


def same_groups(grouped):
    names = list(grouped)
    ref = grouped[names[0]]
    for name in names[1:]:
        other = grouped[name]
        if len(other["ndcg3"]) != len(ref["ndcg3"]):
            raise RuntimeError(f"у {name} другое число групп")
        if not np.array_equal(other["user_id"], ref["user_id"]) or not np.array_equal(other["day"], ref["day"]):
            raise RuntimeError(f"группы {name} не совпали")
    return len(ref["ndcg3"])


def coverage():
    """Доля NULL и доля нулей на val. Test не читается."""
    quoted = ", ".join(f'"{name}"' for name in NEW_FEATURES)
    con = duckdb.connect()
    frame = con.sql(f"""
        SELECT {quoted}
        FROM '{CANDIDATES_PATH}' c
        JOIN '{EXAMPLES_PATH}' e USING (user_id, item_id, day)
        WHERE e.split = 'val'
    """).df()
    rows = []
    for name in NEW_FEATURES:
        values = frame[name]
        rows.append({
            "record": "coverage",
            "name": name,
            "n_rows": len(values),
            "null_share": float(values.isna().mean()),
            "zero_share": float((values == 0).mean()),
        })
    return pd.DataFrame(rows)


def selection_diff():
    table = pd.read_csv(SELECTION_PATH)
    passed = set(table.loc[table["прошёл"] == 1, "признак"])
    threshold = float(table["сила"].mean())
    rows = []
    for name in NEW_FEATURES:
        part = table.loc[table["признак"] == name].iloc[0]
        rows.append({
            "признак": name,
            "group_auc": float(part["group_auc"]),
            "сила": float(part["сила"]),
            "прошёл": int(part["прошёл"]),
            "причина": part["причина"],
            "изменение": "новый",
            "порог": threshold,
        })
    for name in A_OLD:
        if name not in passed:
            part = table.loc[table["признак"] == name].iloc[0]
            rows.append({
                "признак": name,
                "group_auc": float(part["group_auc"]),
                "сила": float(part["сила"]),
                "прошёл": int(part["прошёл"]),
                "причина": part["причина"],
                "изменение": "выбыл из A",
                "порог": threshold,
            })
    for name in sorted(passed - set(A_OLD) - set(NEW_FEATURES)):
        part = table.loc[table["признак"] == name].iloc[0]
        rows.append({
            "признак": name,
            "group_auc": float(part["group_auc"]),
            "сила": float(part["сила"]),
            "прошёл": int(part["прошёл"]),
            "причина": part["причина"],
            "изменение": "вошёл в A",
            "порог": threshold,
        })
    return pd.DataFrame(rows), threshold, passed


def main():
    all_features = candidate_names()
    sets = model_sets(all_features, selected_names())
    frame = load_modeling_frame(all_features)
    train = frame[frame["split"] == "train"].reset_index(drop=True)
    val = frame[frame["split"] == "val"].reset_index(drop=True)
    ensure_train_rows(train)
    ensure_val_rows(val)
    popularity, _prior_views = baseline_scores(val)

    grouped = {}
    metric_rows = []
    for name in ("A_old", "A_new", "A_old_personal", "A_old_category", "A_old_cat", "C_new"):
        features = sets[name]
        print(f"обучаю {name}: {len(features)} признаков", flush=True)
        model, scores = fit_model(train, val, features)
        grouped[name] = per_group_metrics(val, scores, seed=TIE_SEED)
        auc = float(roc_auc_score(val["label"], scores))
        info = grouped[name]
        metric_rows.append({
            "record": "metrics",
            "name": name,
            "n_features": len(features),
            "n_groups": len(info["ndcg3"]),
            "best_iteration": int(model.best_iteration_ or 0),
            "ndcg_3": float(info["ndcg3"].mean()),
            "ndcg_5": float(info["ndcg5"].mean()),
            "mrr": float(info["mrr"].mean()),
            "auc": auc,
        })
        print(
            f"  NDCG@3 {metric_rows[-1]['ndcg_3']:.6f}  "
            f"NDCG@5 {metric_rows[-1]['ndcg_5']:.6f}  "
            f"MRR {metric_rows[-1]['mrr']:.6f}  AUC {auc:.6f}",
            flush=True,
        )

    grouped["popularity"] = per_group_metrics(val, popularity, seed=0)
    n_groups = same_groups(grouped)
    if n_groups != 617:
        raise RuntimeError(f"ожидались 617 групп val, получилось {n_groups}")
    indices = bootstrap_indices(n_groups, N_BOOTSTRAP, BOOTSTRAP_SEED)
    diff_rows = []
    for left, right, name in PAIRS:
        estimate, low, high = diff_ci(grouped[left]["ndcg3"], grouped[right]["ndcg3"], indices)
        diff_rows.append({
            "record": "diff",
            "name": name,
            "n_groups": n_groups,
            "ndcg_3": estimate,
            "ndcg_3_low": low,
            "ndcg_3_high": high,
        })
    metrics = pd.DataFrame(metric_rows)
    diffs = pd.DataFrame(diff_rows)
    cover = coverage()
    report = pd.concat([metrics, diffs, cover], ignore_index=True)
    report.to_csv(OUT_PATH, index=False)
    changed, threshold, passed = selection_diff()
    changed.to_csv(SELECTION_DIFF_PATH, index=False)
    print(f"групп: {n_groups}")
    print(f"popularity NDCG@3 {grouped['popularity']['ndcg3'].mean():.6f}")
    print(f"порог: {threshold:.6f}")
    print(f"в A_new: {len(passed)}")
    print(diffs.to_string(index=False))
    print(cover.to_string(index=False))
    print(changed.to_string(index=False))


if __name__ == "__main__":
    main()
