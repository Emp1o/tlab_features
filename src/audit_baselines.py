"""Аудит бейзлайнов на val1 и val2. Test не читается и не оценивается.

Ручные 27 признаков и полный пул C (152) замораживаются числом деревьев:
медиана best_iteration на val1 по сидам 42–46. A_new и A_new+7 остаются
на уже выбранных 80 деревьях. Ранняя остановка на оцениваемом окне — только
как справка и как источник медианы для manual и C.

Замер генерации пишет в /tmp и отбрасывает строки test. Официальные parquet
и report/feature_selection.csv не перезаписываются.
"""

import gc
import pickle
import threading
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from feature_generator import main as generate_candidates
from feature_selector import REPORT_PATH, run_selection
from features_manual import MANUAL_FEATURES, main as generate_manual
from second_period import (
    A_NEW,
    A_NEW_PLUS,
    BASELINE_TIE_SEED,
    SEEDS,
    VAL1_FIRST,
    VAL1_LAST,
    VAL2_FIRST,
    VAL2_LAST,
    baseline_scores,
    metric_row,
    slice_period,
)
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
    candidate_names,
    diff_ci,
    per_group_metrics,
    same_groups,
)

MANUAL_PATH = ROOT / "data" / "features_manual.parquet"
OUT_PATH = ROOT / "report" / "audit_baselines.csv"
COST_PATH = ROOT / "report" / "cost.csv"
STATE_PATH = Path("/tmp/audit_baselines_state.pkl")
CANDIDATES_TMP = "/tmp/audit_candidates.parquet"
MANUAL_TMP = "/tmp/audit_manual.parquet"
SELECTION_TMP = "/tmp/audit_feature_selection.csv"
LAST_ALLOWED_DAY = "2015-09-03"
EXAMPLES_WHERE = f"split IN ('train', 'val') AND day <= DATE '{LAST_ALLOWED_DAY}'"
# Уже выбранная на val1 медиана A_new. Заново не выбирается.
A_NEW_ROUNDS = 80
A_PLUS_ROUNDS = 80
PUBLISHED = {
    ("val2", "A_new", "early"): 0.6766763790560388,
    ("val2", "A_new", "fixed"): 0.6792409009398029,
    ("val1", "A_new", "early"): 0.6580,
}


def rss_mb():
    with open("/proc/self/status", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return float("nan")


class RssWatch:
    def __init__(self):
        self.peak = rss_mb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop.wait(0.2):
            value = rss_mb()
            if value > self.peak:
                self.peak = value

    def close(self):
        self.peak = max(self.peak, rss_mb())
        self._stop.set()
        self._thread.join(timeout=1)
        return self.peak


def empty_state():
    return {
        "done": [],
        "metrics": [],
        "diffs": [],
        "seeds": [],
        "rounds": [],
        "costs": [],
        "groups": {},
        "indices": {},
    }


def load_state():
    if STATE_PATH.exists():
        with STATE_PATH.open("rb") as handle:
            return pickle.load(handle)
    return empty_state()


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE_PATH.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(STATE_PATH)
    write_reports(state)


def write_reports(state):
    frames = []
    for key in ("rounds", "metrics", "diffs", "seeds"):
        if state[key]:
            frames.append(pd.DataFrame(state[key]))
    report = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT_PATH.with_suffix(".csv.tmp")
    report.to_csv(temporary, index=False)
    temporary.replace(OUT_PATH)
    cost = pd.DataFrame(state["costs"])
    temporary = COST_PATH.with_suffix(".csv.tmp")
    cost.to_csv(temporary, index=False)
    temporary.replace(COST_PATH)


def assert_no_test(df, what):
    if df["split"].eq("test").any():
        raise RuntimeError(f"{what}: в кадре есть split=test")
    if pd.to_datetime(df["day"]).max() > pd.Timestamp(LAST_ALLOWED_DAY):
        raise RuntimeError(f"{what}: есть день после {LAST_ALLOWED_DAY}")


def median_iteration(iterations):
    ordered = sorted(int(item) for item in iterations)
    return int(ordered[len(ordered) // 2])


def group_key(period, name, protocol):
    return f"{period}|{name}|{protocol}"


def check_published(period, name, protocol, ndcg):
    expected = PUBLISHED.get((period, name, protocol))
    if expected is None:
        return
    gap = abs(ndcg - expected)
    print(f"  сверка с опубликованным NDCG@3 {expected:.6f}: разница {gap:.6g}", flush=True)
    if gap > 0.005:
        raise RuntimeError(f"{period} {name} {protocol}: NDCG@3 {ndcg} далеко от {expected}")


def fit_seed_timed(train, val, features, seed, n_rounds):
    """То же разделение, что fit_seed: ранняя остановка либо ровно n_rounds без меток val."""
    columns = list(features)
    params = dict(PARAMS)
    params["random_state"] = seed
    labels = train["label"].to_numpy()
    watch = RssWatch()
    started = time.perf_counter()
    if n_rounds is None:
        model = lgb.LGBMClassifier(**params)
        model.fit(
            train[columns],
            labels,
            eval_X=val[columns],
            eval_y=val["label"].to_numpy(),
            callbacks=[
                lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        iteration = int(model.best_iteration_ or 0)
        trained = time.perf_counter()
        scores = model.predict_proba(val[columns], num_iteration=iteration)[:, 1]
    else:
        params["n_estimators"] = int(n_rounds)
        model = lgb.LGBMClassifier(**params)
        model.fit(train[columns], labels)
        iteration = int(n_rounds)
        trained = time.perf_counter()
        scores = model.predict_proba(val[columns], num_iteration=iteration)[:, 1]
    inferred = time.perf_counter()
    peak = watch.close()
    del model
    return scores, iteration, trained - started, inferred - trained, peak


def run_model(state, period, name, protocol, train, val, features, n_rounds):
    key = group_key(period, name, protocol)
    if key in state["done"]:
        print(f"уже есть {key}", flush=True)
        return
    columns = sorted(features)
    print(
        f"{period} {name} {protocol}: {len(columns)} признаков, деревьев {n_rounds or 'early'}",
        flush=True,
    )
    assert_no_test(train, f"train {key}")
    assert_no_test(val, f"val {key}")
    total = None
    iterations = []
    seed_ndcg = []
    for seed in SEEDS:
        scores, iteration, train_s, infer_s, peak = fit_seed_timed(
            train, val, columns, seed, n_rounds
        )
        grouped = per_group_metrics(val, scores, seed=TIE_SEED)
        seed_ndcg.append(float(grouped["ndcg3"].mean()))
        iterations.append(iteration)
        total = scores if total is None else total + scores
        state["costs"].append({
            "record": "cost",
            "stage": "train",
            "name": name,
            "period": period,
            "protocol": protocol,
            "seed": seed,
            "train_seconds": train_s,
            "infer_seconds": infer_s,
            "peak_ram_mb": peak,
            "n_rows": len(train),
            "n_features": len(columns),
            "note": "infer_seconds — финальный predict_proba; у early train_seconds включает проходы остановки",
        })
        print(
            f"  сид {seed}: деревьев {iteration}, обучение {train_s:.1f} с, "
            f"вывод {infer_s:.1f} с, RSS {peak:.0f} МБ, NDCG@3 {seed_ndcg[-1]:.6f}",
            flush=True,
        )
        del scores, grouped
    averaged = total / len(SEEDS)
    row, grouped = metric_row(period, name, val, averaged, TIE_SEED, len(columns), iterations)
    row["protocol"] = protocol
    row["rounds"] = "" if n_rounds is None else int(n_rounds)
    state["metrics"].append(row)
    state["seeds"].append({
        "record": "seeds",
        "period": period,
        "name": name,
        "protocol": protocol,
        "n_features": len(columns),
        "ndcg_3_min": min(seed_ndcg),
        "ndcg_3_max": max(seed_ndcg),
        "ndcg_3_seeds": ",".join(f"{value:.6f}" for value in seed_ndcg),
        "iterations": ",".join(str(item) for item in iterations),
    })
    state["groups"][key] = {
        "ndcg3": grouped["ndcg3"],
        "user_id": grouped["user_id"],
        "day": np.asarray(grouped["day"]),
    }
    check_published(period, name, protocol, row["ndcg_3"])
    print(
        f"  среднее NDCG@3 {row['ndcg_3']:.6f}  recall@3 {row['recall_3']:.6f}  "
        f"logloss {row['logloss']:.6f}",
        flush=True,
    )
    if protocol == "early":
        note = "медиана best_iteration по сидам 42–46"
        if name in ("A_new", "A_new_plus"):
            note += "; число деревьев fixed остаётся 80"
        elif period == "val1" and name in ("manual", "C"):
            note += "; это число деревьев для fixed на val1 и val2, без eval_set"
        else:
            note += "; справка, fixed этим числом не задаётся"
        state["rounds"].append({
            "record": "rounds",
            "period": period,
            "name": name,
            "protocol": "early_iterations",
            "rounds": median_iteration(iterations),
            "iterations": ",".join(str(item) for item in iterations),
            "n_features": len(columns),
            "note": note,
        })
    expected_groups = {"val1": 617, "val2": 589}
    if row["n_groups"] != expected_groups[period]:
        raise RuntimeError(f"{key}: групп {row['n_groups']}, ожидалось {expected_groups[period]}")
    state["done"].append(key)
    save_state(state)
    del averaged, total
    gc.collect()


def run_popularity(state, period, val, event_last, manual_split):
    key = group_key(period, "popularity", "baseline")
    if key in state["done"]:
        print(f"уже есть {key}", flush=True)
        return
    assert_no_test(val, f"val {key}")
    watch = RssWatch()
    started = time.perf_counter()
    popularity, _, _ = baseline_scores(val, event_last, manual_split)
    elapsed = time.perf_counter() - started
    peak = watch.close()
    row, grouped = metric_row(
        period, "popularity", val, popularity, BASELINE_TIE_SEED, None, []
    )
    row["protocol"] = "baseline"
    row["rounds"] = ""
    # Счётчик, даже если все значения случайно попали в [0, 1].
    row["logloss"] = float("nan")
    state["metrics"].append(row)
    state["groups"][key] = {
        "ndcg3": grouped["ndcg3"],
        "user_id": grouped["user_id"],
        "day": np.asarray(grouped["day"]),
    }
    state["costs"].append({
        "record": "cost",
        "stage": "baseline",
        "name": "popularity",
        "period": period,
        "protocol": "baseline",
        "seed": "",
        "train_seconds": 0.0,
        "infer_seconds": elapsed,
        "peak_ram_mb": peak,
        "n_rows": len(val),
        "n_features": 1,
        "note": f"корзины и покупки товара за 7 дней, события не позже {event_last}",
    })
    expected_groups = {"val1": 617, "val2": 589}
    if row["n_groups"] != expected_groups[period]:
        raise RuntimeError(f"{key}: групп {row['n_groups']}, ожидалось {expected_groups[period]}")
    state["done"].append(key)
    save_state(state)
    print(f"{period} popularity NDCG@3 {row['ndcg_3']:.6f}  logloss {row['logloss']}", flush=True)


def remember_indices(state, period, reference_key):
    grouped = state["groups"][reference_key]
    n_groups = len(grouped["ndcg3"])
    if period not in state["indices"]:
        state["indices"][period] = bootstrap_indices(n_groups, N_BOOTSTRAP, BOOTSTRAP_SEED)
    if len(state["indices"][period]) != N_BOOTSTRAP or state["indices"][period].shape[1] != n_groups:
        raise RuntimeError(f"{period}: индексы бутстрепа не совпали с числом групп")
    return state["indices"][period]


def add_diff(state, period, left_key, right_key, name):
    if any(row["record"] == "diff" and row["period"] == period and row["name"] == name for row in state["diffs"]):
        return
    left = state["groups"][left_key]
    right = state["groups"][right_key]
    if not same_groups(left, right):
        raise RuntimeError(f"{period} {name}: группы (user_id, day) разошлись")
    indices = remember_indices(state, period, left_key)
    estimate, low, high = diff_ci(left["ndcg3"], right["ndcg3"], indices)
    state["diffs"].append({
        "record": "diff",
        "period": period,
        "name": name,
        "n_groups": len(left["ndcg3"]),
        "ndcg_3": estimate,
        "ndcg_3_low": low,
        "ndcg_3_high": high,
        "note": "парный бутстреп 1000, seed 42, одни индексы внутри периода",
    })
    print(f"  {name}: {estimate:.6f} ({low:.6f}–{high:.6f})", flush=True)


def small_diffs(state, period):
    mark = f"diffs|{period}|small"
    if mark in state["done"]:
        return
    add_diff(state, period, group_key(period, "manual", "early"), group_key(period, "A_new", "early"), "manual_early-A_new_early")
    add_diff(state, period, group_key(period, "manual", "early"), group_key(period, "popularity", "baseline"), "manual_early-popularity")
    add_diff(state, period, group_key(period, "manual", "fixed"), group_key(period, "A_new", "fixed"), "manual_fixed-A_new_fixed")
    add_diff(state, period, group_key(period, "manual", "fixed"), group_key(period, "popularity", "baseline"), "manual_fixed-popularity")
    add_diff(state, period, group_key(period, "A_new", "early"), group_key(period, "popularity", "baseline"), "A_new_early-popularity")
    add_diff(state, period, group_key(period, "A_new", "fixed"), group_key(period, "popularity", "baseline"), "A_new_fixed-popularity")
    add_diff(state, period, group_key(period, "A_new_plus", "early"), group_key(period, "A_new", "early"), "A_new_plus_early-A_new_early")
    add_diff(state, period, group_key(period, "A_new_plus", "fixed"), group_key(period, "A_new", "fixed"), "A_new_plus_fixed-A_new_fixed")
    state["done"].append(mark)
    save_state(state)


def fixed_rounds(state, name):
    """Медиана best_iteration на val1. Для A_new и A_new+7 это не источник 80."""
    matches = [
        row for row in state["rounds"]
        if row["name"] == name and row["period"] == "val1" and row["protocol"] == "early_iterations"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"нет медианы val1 для {name}")
    return int(matches[0]["rounds"])


def load_small():
    quoted_c = ", ".join(f'c."{name}"::FLOAT AS "{name}"' for name in sorted(set(A_NEW_PLUS)))
    quoted_m = ", ".join(f'm."{name}"::FLOAT AS "{name}"' for name in MANUAL_FEATURES)
    con = duckdb.connect()
    frame = con.sql(f"""
        SELECT e.user_id, e.item_id, e.day, e.label, e.split, {quoted_c}, {quoted_m}
        FROM '{EXAMPLES_PATH}' e
        JOIN '{CANDIDATES_PATH}' c USING (user_id, item_id, day)
        JOIN '{MANUAL_PATH}' m USING (user_id, item_id, day)
        WHERE e.split IN ('train', 'val') AND e.day <= DATE '{LAST_ALLOWED_DAY}'
        ORDER BY e.split, e.user_id, e.day, e.item_id
    """).df()
    assert_no_test(frame, "small")
    expected = con.sql(f"""
        SELECT count(*) FROM '{EXAMPLES_PATH}'
        WHERE split IN ('train', 'val') AND day <= DATE '{LAST_ALLOWED_DAY}'
    """).fetchone()[0]
    if len(frame) != int(expected):
        raise RuntimeError("число строк small не совпало с train+val")
    return frame


def load_period_c(train_last, val_first, val_last, val_split):
    names = candidate_names()
    if len(names) != 152:
        raise RuntimeError(f"в пуле ожидалось 152 признака, есть {len(names)}")
    quoted = ", ".join(f'c."{name}"::FLOAT AS "{name}"' for name in names)
    if val_split == "val":
        val_sql = f"e.split = 'val' AND e.day BETWEEN DATE '{val_first}' AND DATE '{val_last}'"
    else:
        val_sql = (
            f"e.split = 'train' AND e.day BETWEEN DATE '{val_first}' AND DATE '{val_last}'"
        )
    con = duckdb.connect()
    train = con.sql(f"""
        SELECT e.user_id, e.item_id, e.day, e.label, e.split, {quoted}
        FROM '{EXAMPLES_PATH}' e
        JOIN '{CANDIDATES_PATH}' c USING (user_id, item_id, day)
        WHERE e.split = 'train' AND e.day <= DATE '{train_last}'
        ORDER BY e.user_id, e.day, e.item_id
    """).df()
    val = con.sql(f"""
        SELECT e.user_id, e.item_id, e.day, e.label, e.split, {quoted}
        FROM '{EXAMPLES_PATH}' e
        JOIN '{CANDIDATES_PATH}' c USING (user_id, item_id, day)
        WHERE {val_sql}
        ORDER BY e.user_id, e.day, e.item_id
    """).df()
    assert_no_test(train, "C train")
    assert_no_test(val, "C val")
    if pd.to_datetime(train["day"]).max() >= pd.to_datetime(val["day"]).min():
        raise RuntimeError("C: train пересекается с val")
    return train, val, names


def time_call(state, stage, name, note, function):
    key = f"cost|{stage}|{name}"
    if key in state["done"]:
        print(f"уже есть {key}", flush=True)
        return
    watch = RssWatch()
    started = time.perf_counter()
    detail = function()
    elapsed = time.perf_counter() - started
    peak = watch.close()
    state["costs"].append({
        "record": "cost",
        "stage": stage,
        "name": name,
        "period": "",
        "protocol": "",
        "seed": "",
        "train_seconds": elapsed,
        "infer_seconds": 0.0,
        "peak_ram_mb": peak,
        "n_rows": detail.get("n_rows", ""),
        "n_features": detail.get("n_features", ""),
        "note": note + (("; " + detail["note"]) if detail.get("note") else ""),
    })
    state["done"].append(key)
    save_state(state)
    print(f"{stage} {name}: {elapsed:.1f} с, RSS {peak:.0f} МБ", flush=True)
    gc.collect()


def candidates_match(path):
    con = duckdb.connect()
    late = con.sql(f"""
        SELECT count(*) FROM '{path}' t
        JOIN '{EXAMPLES_PATH}' e USING (user_id, item_id, day)
        WHERE e.split = 'test' OR t.day > DATE '{LAST_ALLOWED_DAY}'
    """).fetchone()[0]
    if int(late) != 0:
        raise RuntimeError("во временных кандидатах есть test или день после val1")
    mismatch = con.sql(f"""
        SELECT count(*) FROM (
            SELECT c.user_id, c.item_id, c.day,
                   c.user__view__7__count,
                   c.item__transaction__30__count,
                   c.user_item__view__1__count,
                   c.item__available__asof,
                   c.user_category__transaction__30__count,
                   c.category_all__transaction__repeat_share
            FROM '{CANDIDATES_PATH}' c
            JOIN '{EXAMPLES_PATH}' e USING (user_id, item_id, day)
            WHERE e.split IN ('train', 'val') AND e.day <= DATE '{LAST_ALLOWED_DAY}'
            EXCEPT
            SELECT user_id, item_id, day,
                   user__view__7__count,
                   item__transaction__30__count,
                   user_item__view__1__count,
                   item__available__asof,
                   user_category__transaction__30__count,
                   category_all__transaction__repeat_share
            FROM '{path}'
        )
    """).fetchone()[0]
    n_rows = con.sql(f"SELECT count(*) FROM '{path}'").fetchone()[0]
    n_features = len(con.sql(f"DESCRIBE SELECT * FROM '{path}'").fetchall()) - 3
    if int(mismatch) != 0:
        raise RuntimeError(f"временные кандидаты не совпали с официальными на {mismatch} строках")
    if int(n_features) != 152:
        raise RuntimeError(f"во временном пуле {n_features} признаков, ожидалось 152")
    return {"n_rows": int(n_rows), "n_features": int(n_features), "note": "совпало с официальным parquet на train+val"}


def manual_match(path):
    con = duckdb.connect()
    late = con.sql(f"""
        SELECT count(*) FROM '{path}'
        WHERE split = 'test' OR day > DATE '{LAST_ALLOWED_DAY}'
    """).fetchone()[0]
    if int(late) != 0:
        raise RuntimeError("во временных ручных признаках есть test или день после val1")
    described = [row[0] for row in con.sql(f"DESCRIBE SELECT * FROM '{path}'").fetchall()]
    missing = [name for name in MANUAL_FEATURES if name not in described]
    if missing:
        raise RuntimeError(f"нет ручных признаков: {missing}")
    mismatch = con.sql(f"""
        SELECT count(*) FROM (
            SELECT user_id, item_id, day, user_views_7d, ui_views_30d, uc_buys_30d, cat_views_7d
            FROM '{MANUAL_PATH}'
            WHERE split IN ('train', 'val') AND day <= DATE '{LAST_ALLOWED_DAY}'
            EXCEPT
            SELECT user_id, item_id, day, user_views_7d, ui_views_30d, uc_buys_30d, cat_views_7d
            FROM '{path}'
        )
    """).fetchone()[0]
    n_rows = con.sql(f"SELECT count(*) FROM '{path}'").fetchone()[0]
    if int(mismatch) != 0:
        raise RuntimeError(f"временные ручные признаки не совпали с официальными на {mismatch} строках")
    return {"n_rows": int(n_rows), "n_features": len(MANUAL_FEATURES), "note": "совпало с официальным parquet на train+val"}


def generate_candidates_cost():
    generate_candidates(CANDIDATES_TMP, EXAMPLES_WHERE, LAST_ALLOWED_DAY)
    detail = candidates_match(CANDIDATES_TMP)
    Path(CANDIDATES_TMP).unlink(missing_ok=True)
    return detail


def generate_manual_cost():
    generate_manual(MANUAL_TMP, EXAMPLES_WHERE, LAST_ALLOWED_DAY)
    detail = manual_match(MANUAL_TMP)
    Path(MANUAL_TMP).unlink(missing_ok=True)
    return detail


def selection_cost():
    if Path(SELECTION_TMP).resolve() == Path(REPORT_PATH).resolve():
        raise RuntimeError("замер отбора смотрит в официальный csv")
    table, threshold = run_selection(None, SELECTION_TMP)
    passed = set(table.loc[table["прошёл"] == 1, "признак"])
    note = f"порог {threshold:.6f}, прошло {len(passed)}"
    if passed != set(A_NEW):
        note += "; набор не совпал с A_new, официальный список не менялся"
        print("ВНИМАНИЕ:", note, flush=True)
    else:
        note += "; совпал с A_new, официальный csv не переписан"
    return {"n_rows": len(table), "n_features": len(passed), "note": note}


def run_small_period(state, frame, period, train_last, val_first, val_last, val_split, event_last, manual_split):
    if f"diffs|{period}|small" in state["done"]:
        print(f"уже есть малые модели {period}", flush=True)
        return
    train, val = slice_period(frame, train_last, val_first, val_last, val_split)
    assert_no_test(train, period)
    assert_no_test(val, period)
    print(f"{period}: train {len(train)}  val {len(val)}", flush=True)
    run_model(state, period, "A_new", "early", train, val, A_NEW, None)
    run_model(state, period, "A_new", "fixed", train, val, A_NEW, A_NEW_ROUNDS)
    run_model(state, period, "A_new_plus", "early", train, val, A_NEW_PLUS, None)
    run_model(state, period, "A_new_plus", "fixed", train, val, A_NEW_PLUS, A_PLUS_ROUNDS)
    run_model(state, period, "manual", "early", train, val, MANUAL_FEATURES, None)
    manual_rounds = fixed_rounds(state, "manual") if period == "val1" else fixed_rounds(state, "manual")
    if period == "val2" and manual_rounds != fixed_rounds(state, "manual"):
        raise RuntimeError("медиана manual на val2 перевыбирается")
    run_model(state, period, "manual", "fixed", train, val, MANUAL_FEATURES, manual_rounds)
    run_popularity(state, period, val, event_last, manual_split)
    small_diffs(state, period)
    del train, val
    gc.collect()


def run_c(state):
    if "diffs|C" in state["done"]:
        print("уже есть C", flush=True)
        return
    if group_key("val1", "C", "early") not in state["done"] or group_key("val1", "C", "fixed") not in state["done"]:
        train, val, names = load_period_c("2015-08-20", VAL1_FIRST, VAL1_LAST, "val")
        print(f"C val1: train {len(train)}  val {len(val)}", flush=True)
        run_model(state, "val1", "C", "early", train, val, names, None)
        run_model(state, "val1", "C", "fixed", train, val, names, fixed_rounds(state, "C"))
        del train, val
        gc.collect()
    if group_key("val2", "C", "fixed") not in state["done"]:
        train, val, names = load_period_c("2015-08-06", VAL2_FIRST, VAL2_LAST, "train")
        print(f"C val2: train {len(train)}  val {len(val)}", flush=True)
        run_model(state, "val2", "C", "fixed", train, val, names, fixed_rounds(state, "C"))
        del train, val
        gc.collect()
    add_diff(state, "val1", group_key("val1", "C", "fixed"), group_key("val1", "A_new", "fixed"), "C_fixed-A_new_fixed")
    add_diff(state, "val1", group_key("val1", "C", "early"), group_key("val1", "A_new", "early"), "C_early-A_new_early")
    add_diff(state, "val2", group_key("val2", "C", "fixed"), group_key("val2", "A_new", "fixed"), "C_fixed-A_new_fixed")
    state["done"].append("diffs|C")
    save_state(state)


def main():
    if len(MANUAL_FEATURES) != 27:
        raise RuntimeError("ручных признаков не 27")
    state = load_state()
    time_call(
        state,
        "feature_generation",
        "candidates",
        "train+val, day<=2015-09-03, файл /tmp, официальный parquet не переписан",
        generate_candidates_cost,
    )
    time_call(
        state,
        "feature_generation",
        "manual",
        "27 ручных признаков, train+val, day<=2015-09-03, файл /tmp",
        generate_manual_cost,
    )
    time_call(
        state,
        "selection",
        "selector_v2",
        "тот же отборщик на split=train, отчёт в /tmp, A_new не заменяется",
        selection_cost,
    )
    need_small = any(
        f"diffs|{period}|small" not in state["done"] for period in ("val1", "val2")
    )
    if need_small:
        frame = load_small()
        run_small_period(
            state, frame, "val1", "2015-08-20", VAL1_FIRST, VAL1_LAST, "val", VAL1_LAST, "val"
        )
        run_small_period(
            state, frame, "val2", "2015-08-06", VAL2_FIRST, VAL2_LAST, "train", VAL2_LAST, "train"
        )
        del frame
        gc.collect()
    run_c(state)
    manual_rounds = fixed_rounds(state, "manual")
    c_rounds = fixed_rounds(state, "C")
    print(f"фиксированные деревья: manual {manual_rounds}, C {c_rounds}, A_new {A_NEW_ROUNDS}", flush=True)
    print(f"записано {OUT_PATH}", flush=True)
    print(f"записано {COST_PATH}", flush=True)


if __name__ == "__main__":
    main()
