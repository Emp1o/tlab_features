"""Проверка Судьи. Test не читается, решение A_new не меняется.

1. Счётчик короткой истории больше не тождественный ноль.
2. Тот же отборщик на train2 (строго до val2), отдельный csv.
3. Val2: A_new как есть, набор train2, popularity; одни бутстреп-индексы.
4. Число деревьев — медиана best_iteration A_new на val1 по сидам 42–46.
   На val2 A_new и A_new+7 обучаются ровно на это число, без меток val2.
"""

import numpy as np
import pandas as pd

from feature_selector import run_selection
from second_period import (
    A_NEW,
    A_NEW_PLUS,
    TRAIN2_LAST,
    VAL1_FIRST,
    VAL1_LAST,
    VAL2_LAST,
    average_scores,
    baseline_scores,
    history_note,
    load_frame,
    metric_row,
    slice_period,
)
from train_lgbm import BOOTSTRAP_SEED, N_BOOTSTRAP, ROOT, TIE_SEED, bootstrap_indices, diff_ci

TRAIN2_SELECTION = ROOT / "report" / "feature_selection_train2.csv"
OUT_PATH = ROOT / "report" / "judge_review.csv"


def median_iteration(iterations):
    """Средний элемент отсортированного списка. Сидов пять, медиана — целое."""
    ordered = sorted(int(item) for item in iterations)
    return ordered[len(ordered) // 2]


def compare_sets(selected):
    frozen = set(A_NEW)
    chosen = set(selected)
    union = frozen | chosen
    inter = frozen & chosen
    return {
        "dropped": sorted(frozen - chosen),
        "added": sorted(chosen - frozen),
        "jaccard": len(inter) / len(union),
        "n_inter": len(inter),
        "n_union": len(union),
    }


def main():
    train_short, val_short, first_day = history_note()
    print(
        f"первое событие до конца train2: {first_day}; "
        f"train2 короче 30 дней: {train_short}; val2 короче 30 дней: {val_short}",
        flush=True,
    )

    table, threshold = run_selection(TRAIN2_LAST, TRAIN2_SELECTION)
    selected = table.loc[table["прошёл"] == 1, "признак"].tolist()
    compared = compare_sets(selected)
    print(f"порог train2: {threshold:.6f}", flush=True)
    print(f"train2 прошло {len(selected)}: {', '.join(selected)}", flush=True)
    print(f"ушли из A_new: {compared['dropped'] or '—'}", flush=True)
    print(f"пришли к A_new: {compared['added'] or '—'}", flush=True)
    print(f"Jaccard: {compared['jaccard']:.6f}", flush=True)

    frame = load_frame(selected)
    train1, val1 = slice_period(frame, "2015-08-20", VAL1_FIRST, VAL1_LAST, "val")
    print(f"val1 для числа деревьев: train {len(train1)}  val {len(val1)}", flush=True)
    _, val1_seed_ndcg, val1_iterations = average_scores(train1, val1, A_NEW)
    n_rounds = median_iteration(val1_iterations)
    print(f"best_iteration val1 A_new: {val1_iterations}; медиана {n_rounds}", flush=True)
    del train1, val1

    train2, val2 = slice_period(frame, TRAIN2_LAST, "2015-08-07", VAL2_LAST, "train")
    print(f"val2: train {len(train2)}  val {len(val2)}", flush=True)
    popularity, _, _ = baseline_scores(val2, VAL2_LAST, "train")

    jobs = [
        ("early_stop", "A_new", A_NEW, None),
        ("early_stop", "train2_selected", tuple(selected), None),
        ("fixed_rounds", "A_new", A_NEW, n_rounds),
        ("fixed_rounds", "A_new_plus", A_NEW_PLUS, n_rounds),
    ]
    # Одинаковый набор и одинаковый режим не обучаем второй раз.
    fitted_groups = {}
    metric_rows = []
    seed_rows = []
    cache = {}
    for protocol, name, features, rounds in jobs:
        key = (protocol, tuple(sorted(features)), rounds)
        if key not in cache:
            label = f"{protocol}/{name}"
            print(f"обучаю {label}: {len(features)} признаков, rounds={rounds}", flush=True)
            scores, seed_ndcg, iterations = average_scores(train2, val2, features, n_rounds=rounds)
            cache[key] = (scores, seed_ndcg, iterations)
        else:
            print(f"тот же fit, что уже посчитан: {protocol} {name}", flush=True)
            scores, seed_ndcg, iterations = cache[key]
        row, grouped = metric_row("val2", name, val2, scores, TIE_SEED, len(features), iterations)
        row["protocol"] = protocol
        metric_rows.append(row)
        seed_rows.append({
            "record": "seeds",
            "period": "val2",
            "protocol": protocol,
            "name": name,
            "ndcg_3_min": min(seed_ndcg),
            "ndcg_3_max": max(seed_ndcg),
            "ndcg_3_seeds": ",".join(f"{value:.6f}" for value in seed_ndcg),
            "iterations": ",".join(str(item) for item in iterations),
        })
        fitted_groups[(protocol, name)] = grouped
        print(f"  NDCG@3 {row['ndcg_3']:.6f}", flush=True)

    pop_row, pop_grouped = metric_row("val2", "popularity", val2, popularity, 0, None, [])
    pop_row["protocol"] = "baseline"
    metric_rows.append(pop_row)
    fitted_groups[("baseline", "popularity")] = pop_grouped

    anchor = fitted_groups[("early_stop", "A_new")]
    n_groups = len(anchor["ndcg3"])
    for grouped in fitted_groups.values():
        if len(grouped["ndcg3"]) != n_groups:
            raise RuntimeError("разное число групп")
        if not np.array_equal(grouped["user_id"], anchor["user_id"]):
            raise RuntimeError("разные группы")
        if not np.array_equal(grouped["day"], anchor["day"]):
            raise RuntimeError("разные дни групп")
    indices = bootstrap_indices(n_groups, N_BOOTSTRAP, BOOTSTRAP_SEED)
    pairs = (
        ("early_stop", "train2_selected", "early_stop", "A_new", "train2_selected-A_new"),
        ("early_stop", "A_new", "baseline", "popularity", "A_new-popularity"),
        ("early_stop", "train2_selected", "baseline", "popularity", "train2_selected-popularity"),
        ("fixed_rounds", "A_new_plus", "fixed_rounds", "A_new", "A_new_plus-A_new"),
        ("fixed_rounds", "A_new", "baseline", "popularity", "A_new-popularity"),
        ("fixed_rounds", "A_new_plus", "baseline", "popularity", "A_new_plus-popularity"),
    )
    diff_rows = []
    for left_p, left_n, right_p, right_n, name in pairs:
        estimate, low, high = diff_ci(
            fitted_groups[(left_p, left_n)]["ndcg3"],
            fitted_groups[(right_p, right_n)]["ndcg3"],
            indices,
        )
        protocol = left_p if left_p == right_p else f"{left_p}_vs_{right_p}"
        diff_rows.append({
            "record": "diff",
            "period": "val2",
            "protocol": protocol,
            "name": name,
            "n_groups": n_groups,
            "ndcg_3": estimate,
            "ndcg_3_low": low,
            "ndcg_3_high": high,
        })
        print(f"{protocol} {name}: {estimate:.6f} ({low:.6f}–{high:.6f})", flush=True)

    rows = []
    rows.append({
        "record": "history",
        "period": "train2",
        "name": "rows_before_30d",
        "n_groups": train_short,
        "note": f"first_event_day={first_day}",
    })
    rows.append({
        "record": "history",
        "period": "val2",
        "name": "rows_before_30d",
        "n_groups": val_short,
        "note": f"first_event_day={first_day}",
    })
    rows.append({
        "record": "selection",
        "period": "train2",
        "name": "threshold",
        "ndcg_3": threshold,
        "n_features": len(selected),
        "note": ",".join(selected),
    })
    rows.append({
        "record": "selection",
        "period": "train2",
        "name": "jaccard_vs_A_new",
        "ndcg_3": compared["jaccard"],
        "n_features": compared["n_inter"],
        "n_groups": compared["n_union"],
        "note": "dropped=" + ",".join(compared["dropped"]) + "; added=" + ",".join(compared["added"]),
    })
    rows.append({
        "record": "rounds",
        "period": "val1",
        "name": "A_new_median_best_iteration",
        "n_features": n_rounds,
        "iterations": ",".join(str(item) for item in val1_iterations),
        "ndcg_3_seeds": ",".join(f"{value:.6f}" for value in val1_seed_ndcg),
        "note": "медиана best_iteration, сиды 42-46, ранняя остановка на val1",
    })
    for row in metric_rows:
        row["record"] = "metrics"
        rows.append(row)
    rows.extend(diff_rows)
    rows.extend(seed_rows)
    report = pd.DataFrame(rows)
    report.to_csv(OUT_PATH, index=False)
    print(f"записано {OUT_PATH}", flush=True)
    print(f"фиксированных деревьев: {n_rounds}", flush=True)


if __name__ == "__main__":
    main()
