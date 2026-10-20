"""Автопоиск окон по предрегистрации в STATUS.md.

Сетка W = {1, 3, 5, 7, 10, 14, 21, 30}. W_auto отбирается на train2.
Правило на val2 сравнивает его с A_new. Test здесь не оценивается:
файл признаков может содержать строки test, метрики по ним не считаются.
"""

import gc
import pickle
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from audit_baselines import RssWatch, assert_no_test, fit_seed_timed, median_iteration
from feature_generator import SEARCH_WINDOWS, main as generate_candidates
from feature_selector import run_selection
from features_manual import MANUAL_FEATURES
from second_period import (
    A_NEW,
    BASELINE_TIE_SEED,
    SEEDS,
    TRAIN2_LAST,
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
    EXAMPLES_PATH,
    N_BOOTSTRAP,
    ROOT,
    TIE_SEED,
    bootstrap_indices,
    candidate_names,
    diff_ci,
    same_groups,
)

WSEARCH_PATH = ROOT / "data" / "feature_candidates_wsearch.parquet"
TRAIN2_SELECTION = ROOT / "report" / "window_selection_train2.csv"
TRAIN_SELECTION = ROOT / "report" / "window_selection_train.csv"
OUT_PATH = ROOT / "report" / "window_search.csv"
COST_PATH = ROOT / "report" / "window_cost.csv"
STATE_PATH = Path("/tmp/window_search_state.pkl")
MANUAL_PATH = ROOT / "data" / "features_manual.parquet"
LAST_ALLOWED_DAY = "2015-09-03"
TRAIN_LAST = "2015-08-20"
A_NEW_ROUNDS = 80
C_ROUNDS = 78
MANUAL_ROUNDS = 89
# Число из аудита: A_new, 80 деревьев, val2. Далеко от него — стоп.
PUBLISHED_VAL2_A_NEW = 0.6792409009398029


def empty_state():
    return {
        "done": [],
        "metrics": [],
        "diffs": [],
        "seeds": [],
        "rounds": [],
        "features": [],
        "windows": [],
        "costs": [],
        "groups": {},
        "indices": {},
        "passed": {},
        "thresholds": {},
    }


def load_state():
    if STATE_PATH.exists():
        with STATE_PATH.open("rb") as handle:
            return pickle.load(handle)
    return empty_state()


def save_state(state):
    temporary = STATE_PATH.with_suffix(".pkl.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(STATE_PATH)
    write_reports(state)


def write_reports(state):
    frames = []
    for key in ("rounds", "features", "windows", "metrics", "diffs", "seeds"):
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


def parse_feature(name):
    entities = ("user_category", "user_item", "category_all", "category", "user", "item")
    entity, rest = "", name
    for item in entities:
        if name.startswith(item + "__"):
            entity = item
            rest = name[len(item) + 2:]
            break
    parts = rest.split("__")
    known_event = {"view", "addtocart", "transaction", "available", "category"}
    if len(parts) >= 3 and parts[0] in known_event:
        return entity, parts[0], parts[1], "__".join(parts[2:])
    if len(parts) == 2:
        return entity, parts[0], "", parts[1]
    return entity, "", "", rest


def passed_names(table):
    return table.loc[table["прошёл"] == 1, "признак"].tolist()


def time_stage(state, stage, name, note, function):
    key = f"cost|{stage}|{name}"
    if key in state["done"]:
        print(f"уже есть {key}", flush=True)
        return state.get("results", {}).get(key)
    watch = RssWatch()
    started = time.perf_counter()
    result = function()
    elapsed = time.perf_counter() - started
    peak = watch.close()
    detail = result if isinstance(result, dict) else {}
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
        "note": note if not detail.get("note") else note + "; " + detail["note"],
    })
    state["done"].append(key)
    state.setdefault("results", {})[key] = result
    save_state(state)
    print(f"{stage} {name}: {elapsed:.1f} с, RSS {peak:.0f} МБ", flush=True)
    gc.collect()
    return result


def generate():
    if WSEARCH_PATH.exists():
        WSEARCH_PATH.unlink()
    generate_candidates(
        str(WSEARCH_PATH),
        windows=SEARCH_WINDOWS,
        memory_limit="12GB",
    )
    con = duckdb.connect()
    described = [row[0] for row in con.sql(f"DESCRIBE SELECT * FROM '{WSEARCH_PATH}'").fetchall()]
    names = [name for name in described if name not in ("user_id", "item_id", "day")]
    n_rows = con.sql(f"SELECT count(*) FROM '{WSEARCH_PATH}'").fetchone()[0]
    n_examples = con.sql(f"SELECT count(*) FROM '{EXAMPLES_PATH}'").fetchone()[0]
    if int(n_rows) != int(n_examples):
        raise RuntimeError(f"строк признаков {n_rows}, примеров {n_examples}")
    if len(names) != 272:
        raise RuntimeError(f"в сетке поиска {len(names)} признаков, ожидалось 272")
    return {"n_rows": int(n_rows), "n_features": len(names), "note": "все сплиты, метрики test не считались"}


def select(last_day, report_path):
    table, threshold = run_selection(last_day, str(report_path), str(WSEARCH_PATH))
    names = passed_names(table)
    return {
        "names": names,
        "threshold": float(threshold),
        "n_rows": len(table),
        "n_features": len(names),
        "note": f"порог {threshold:.6f}, прошло {len(names)}",
    }


def remember_selection(state, key, result):
    state["passed"][key] = list(result["names"])
    state["thresholds"][key] = result["threshold"]
    for name in result["names"]:
        entity, event, window, agg = parse_feature(name)
        state["features"].append({
            "record": "list",
            "period": key,
            "name": "W_auto" if key == "train2" else "W_test",
            "feature": name,
            "entity": entity,
            "event": event,
            "window": window,
            "agg": agg,
        })
    grouped = {}
    for name in result["names"]:
        entity, event, window, agg = parse_feature(name)
        grouped.setdefault((entity, event, agg), []).append(window or "—")
    for (entity, event, agg), windows in sorted(grouped.items()):
        shown = [item for item in windows if item != "—"]
        state["windows"].append({
            "record": "windows",
            "period": key,
            "name": "W_auto" if key == "train2" else "W_test",
            "entity": entity,
            "event": event,
            "agg": agg,
            "window": ",".join(shown),
            "n_features": len(windows),
            "note": "окна, прошедшие отбор в этой тройке",
        })
    counts = {}
    for name in result["names"]:
        _, _, window, _ = parse_feature(name)
        counts[window or "—"] = counts.get(window or "—", 0) + 1
    for window, count in sorted(counts.items(), key=lambda item: (item[0] == "—", item[0])):
        state["windows"].append({
            "record": "window_count",
            "period": key,
            "name": "W_auto" if key == "train2" else "W_test",
            "window": window,
            "n_features": count,
            "note": "сколько прошедших признаков использует это окно",
        })


def jaccard(state):
    if "jaccard" in state["done"]:
        return
    left = set(state["passed"]["train2"])
    right = set(state["passed"]["train"])
    union = left | right
    inter = left & right
    value = len(inter) / len(union) if union else float("nan")
    state["windows"].append({
        "record": "jaccard",
        "period": "train2-train",
        "name": "W_auto-W_test",
        "n_features": len(inter),
        "n_rows": len(union),
        "ndcg_3": value,
        "note": (
            f"Jaccard {value:.6f}, пересечение {len(inter)}, объединение {len(union)}; "
            f"только train2: {', '.join(sorted(left - right)) or '—'}; "
            f"только полный train: {', '.join(sorted(right - left)) or '—'}"
        ),
    })
    state["done"].append("jaccard")
    save_state(state)
    print(f"Jaccard train2/train: {value:.6f} ({len(inter)}/{len(union)})", flush=True)


def load_columns(path, columns, where):
    quoted = ", ".join(f'f."{name}"::FLOAT AS "{name}"' for name in columns)
    con = duckdb.connect()
    frame = con.sql(f"""
        SELECT e.user_id, e.item_id, e.day, e.label, e.split, {quoted}
        FROM '{EXAMPLES_PATH}' e
        JOIN '{path}' f USING (user_id, item_id, day)
        WHERE {where}
        ORDER BY e.user_id, e.day, e.item_id
    """).df()
    assert_no_test(frame, path.name if hasattr(path, "name") else "frame")
    if frame["split"].eq("test").any():
        raise RuntimeError("в оценку попал test")
    return frame


def period_slices(frame, period):
    if period == "val1":
        return slice_period(frame, TRAIN_LAST, VAL1_FIRST, VAL1_LAST, "val")
    if period == "val2":
        return slice_period(frame, TRAIN2_LAST, VAL2_FIRST, VAL2_LAST, "train")
    raise RuntimeError(period)


def gkey(period, name, protocol):
    return f"{period}|{name}|{protocol}"


def run_model(state, period, name, protocol, train, val, features, n_rounds):
    key = gkey(period, name, protocol)
    if key in state["done"]:
        print(f"уже есть {key}", flush=True)
        return
    columns = sorted(features)
    print(f"{period} {name} {protocol}: {len(columns)} признаков, деревьев {n_rounds or 'early'}", flush=True)
    assert_no_test(train, key)
    assert_no_test(val, key)
    total = None
    iterations = []
    seed_ndcg = []
    for seed in SEEDS:
        scores, iteration, train_s, infer_s, peak = fit_seed_timed(train, val, columns, seed, n_rounds)
        from train_lgbm import per_group_metrics
        one = per_group_metrics(val, scores, seed=TIE_SEED)
        seed_ndcg.append(float(one["ndcg3"].mean()))
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
            "note": "infer — финальный predict_proba; у early обучение включает проходы остановки",
        })
        print(
            f"  сид {seed}: деревьев {iteration}, обучение {train_s:.1f} с, "
            f"вывод {infer_s:.1f} с, RSS {peak:.0f} МБ, NDCG@3 {seed_ndcg[-1]:.6f}",
            flush=True,
        )
        del scores, one
    averaged = total / len(SEEDS)
    row, grouped = metric_row(period, name, val, averaged, TIE_SEED, len(columns), iterations)
    row["protocol"] = protocol
    row["rounds"] = "" if n_rounds is None else int(n_rounds)
    expected = {"val1": 617, "val2": 589}[period]
    if row["n_groups"] != expected:
        raise RuntimeError(f"{key}: групп {row['n_groups']}, ожидалось {expected}")
    if period == "val2" and name == "A_new" and protocol == "fixed":
        gap = abs(row["ndcg_3"] - PUBLISHED_VAL2_A_NEW)
        print(f"  сверка A_new val2 80: разница {gap:.6g}", flush=True)
        if gap > 0.005:
            raise RuntimeError("A_new на val2 разошёлся с опубликованным числом")
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
    if protocol == "early":
        state["rounds"].append({
            "record": "rounds",
            "period": period,
            "name": name,
            "protocol": "early_iterations",
            "rounds": median_iteration(iterations),
            "iterations": ",".join(str(item) for item in iterations),
            "n_features": len(columns),
            "note": "медиана best_iteration, сиды 42–46, val1; fixed без eval_set",
        })
    state["done"].append(key)
    save_state(state)
    print(
        f"  среднее NDCG@3 {row['ndcg_3']:.6f}  recall@3 {row['recall_3']:.6f}  logloss {row['logloss']:.6f}",
        flush=True,
    )
    del averaged, total
    gc.collect()


def run_popularity(state, period, val, event_last, manual_split):
    key = gkey(period, "popularity", "baseline")
    if key in state["done"]:
        return
    assert_no_test(val, key)
    watch = RssWatch()
    started = time.perf_counter()
    popularity, _, _ = baseline_scores(val, event_last, manual_split)
    elapsed = time.perf_counter() - started
    peak = watch.close()
    row, grouped = metric_row(period, "popularity", val, popularity, BASELINE_TIE_SEED, None, [])
    row["protocol"] = "baseline"
    row["rounds"] = ""
    row["logloss"] = float("nan")
    expected = {"val1": 617, "val2": 589}[period]
    if row["n_groups"] != expected:
        raise RuntimeError(f"{key}: групп {row['n_groups']}")
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
        "note": f"события не позже {event_last}",
    })
    state["done"].append(key)
    save_state(state)
    print(f"{period} popularity NDCG@3 {row['ndcg_3']:.6f}", flush=True)


def rounds_of(state, name):
    matches = [
        row for row in state["rounds"]
        if row["name"] == name and row["period"] == "val1" and row["protocol"] == "early_iterations"
    ]
    if len(matches) != 1:
        raise RuntimeError(f"нет медианы val1 для {name}")
    return int(matches[0]["rounds"])


def add_diff(state, period, left_name, right_name, protocol, label):
    if any(row.get("name") == label and row.get("period") == period for row in state["diffs"]):
        return
    left = state["groups"][gkey(period, left_name, protocol if left_name != "popularity" else "baseline")]
    right_protocol = protocol if right_name != "popularity" else "baseline"
    right = state["groups"][gkey(period, right_name, right_protocol)]
    if not same_groups(left, right):
        raise RuntimeError(f"{period} {label}: группы разошлись")
    if period not in state["indices"]:
        state["indices"][period] = bootstrap_indices(len(left["ndcg3"]), N_BOOTSTRAP, BOOTSTRAP_SEED)
    estimate, low, high = diff_ci(left["ndcg3"], right["ndcg3"], state["indices"][period])
    state["diffs"].append({
        "record": "diff",
        "period": period,
        "name": label,
        "protocol": protocol,
        "n_groups": len(left["ndcg3"]),
        "ndcg_3": estimate,
        "ndcg_3_low": low,
        "ndcg_3_high": high,
        "note": "парный бутстреп 1000, seed 42, одни индексы внутри периода",
    })
    print(f"  {period} {label}: {estimate:.6f} ({low:.6f}–{high:.6f})", flush=True)


def apply_decision(state):
    if "decision" in state["done"]:
        return
    row = next(item for item in state["diffs"] if item["period"] == "val2" and item["name"] == "W_auto-A_new")
    replace = float(row["ndcg_3_low"]) > 0
    state["diffs"].append({
        "record": "decision",
        "period": "val2",
        "name": "W_auto-A_new",
        "protocol": "fixed",
        "n_groups": row["n_groups"],
        "ndcg_3": row["ndcg_3"],
        "ndcg_3_low": row["ndcg_3_low"],
        "ndcg_3_high": row["ndcg_3_high"],
        "note": (
            "нижняя граница > 0: W_auto заменяет A_new"
            if replace else
            "нижняя граница не > 0: остаётся A_new"
        ),
    })
    state["done"].append("decision")
    save_state(state)
    print(state["diffs"][-1]["note"], flush=True)


def run_fixed_block(state, period, frames, event_last, manual_split):
    train_a, val_a = frames["A_new"]
    train_m, val_m = frames["manual"]
    train_w, val_w = frames["W_auto"]
    run_model(state, period, "A_new", "fixed", train_a, val_a, A_NEW, A_NEW_ROUNDS)
    run_model(state, period, "manual", "fixed", train_m, val_m, MANUAL_FEATURES, MANUAL_ROUNDS)
    run_model(state, period, "W_auto", "fixed", train_w, val_w, state["passed"]["train2"], rounds_of(state, "W_auto"))
    run_popularity(state, period, val_a, event_last, manual_split)
    for label, left, right in (
        ("W_auto-A_new", "W_auto", "A_new"),
        ("W_auto-manual", "W_auto", "manual"),
        ("W_auto-popularity", "W_auto", "popularity"),
        ("A_new-popularity", "A_new", "popularity"),
        ("manual-popularity", "manual", "popularity"),
    ):
        add_diff(state, period, left, right, "fixed", label)
    save_state(state)


def run_c(state, period, rounds):
    if gkey(period, "C", "fixed") in state["done"]:
        return
    if period == "val1":
        train_last, val_first, val_last, val_split = TRAIN_LAST, VAL1_FIRST, VAL1_LAST, "val"
        where_val = f"e.split = 'val' AND e.day BETWEEN DATE '{val_first}' AND DATE '{val_last}'"
    else:
        train_last, val_first, val_last, val_split = TRAIN2_LAST, VAL2_FIRST, VAL2_LAST, "train"
        where_val = (
            f"e.split = 'train' AND e.day BETWEEN DATE '{val_first}' AND DATE '{val_last}'"
        )
    del val_split
    names = candidate_names()
    if len(names) != 152:
        raise RuntimeError(f"C: {len(names)} признаков, ожидалось 152")
    train = load_columns(
        CANDIDATES_PATH,
        names,
        f"e.split = 'train' AND e.day <= DATE '{train_last}'",
    )
    val = load_columns(CANDIDATES_PATH, names, where_val)
    print(f"C {period}: train {len(train)}  val {len(val)}", flush=True)
    run_model(state, period, "C", "fixed", train, val, names, rounds)
    add_diff(state, period, "W_auto", "C", "fixed", "W_auto-C")
    add_diff(state, period, "C", "A_new", "fixed", "C-A_new")
    del train, val
    gc.collect()
    save_state(state)


def main():
    state = load_state()
    time_stage(
        state,
        "feature_generation",
        "wsearch",
        "сетка 1,3,5,7,10,14,21,30, отдельный parquet, официальный файл не переписан",
        generate,
    )
    result = time_stage(
        state,
        "selection",
        "train2",
        "отбор на train2, day<=2015-08-06, официальный csv не переписан",
        lambda: select(TRAIN2_LAST, TRAIN2_SELECTION),
    )
    if "train2" not in state["passed"] and result:
        remember_selection(state, "train2", result)
        save_state(state)
    result = time_stage(
        state,
        "selection",
        "train",
        "отбор на train по 2015-08-20, для модели test, не для правила val2",
        lambda: select(TRAIN_LAST, TRAIN_SELECTION),
    )
    if "train" not in state["passed"] and result:
        remember_selection(state, "train", result)
        save_state(state)
    if "train2" not in state["passed"] or "train" not in state["passed"]:
        raise RuntimeError("нет списков отбора в состоянии")
    jaccard(state)

    where_small = f"e.split IN ('train', 'val') AND e.day <= DATE '{LAST_ALLOWED_DAY}'"
    w_columns = sorted(set(state["passed"]["train2"]) | set(state["passed"]["train"]))
    if gkey("val1", "W_auto", "early") not in state["done"] or "diffs|val1" not in state["done"]:
        w_frame = load_columns(WSEARCH_PATH, w_columns, where_small)
        a_frame = load_columns(CANDIDATES_PATH, list(A_NEW), where_small)
        m_frame = load_columns(MANUAL_PATH, list(MANUAL_FEATURES), where_small)
        train_w, val_w = period_slices(w_frame, "val1")
        run_model(state, "val1", "W_auto", "early", train_w, val_w, state["passed"]["train2"], None)
        if set(state["passed"]["train"]) != set(state["passed"]["train2"]):
            run_model(state, "val1", "W_test", "early", train_w, val_w, state["passed"]["train"], None)
        else:
            if not any(row["name"] == "W_test" and row["record"] == "rounds" for row in state["rounds"]):
                copied = dict(next(row for row in state["rounds"] if row["name"] == "W_auto"))
                copied["name"] = "W_test"
                copied["note"] = "список совпал с W_auto, медиана та же"
                state["rounds"].append(copied)
                save_state(state)
        frames = {
            "A_new": period_slices(a_frame, "val1"),
            "manual": period_slices(m_frame, "val1"),
            "W_auto": (train_w, val_w),
        }
        run_fixed_block(state, "val1", frames, VAL1_LAST, "val")
        state["done"].append("diffs|val1")
        save_state(state)
        del w_frame, a_frame, m_frame, frames
        gc.collect()

    if "diffs|val2" not in state["done"]:
        w_frame = load_columns(WSEARCH_PATH, w_columns, where_small)
        a_frame = load_columns(CANDIDATES_PATH, list(A_NEW), where_small)
        m_frame = load_columns(MANUAL_PATH, list(MANUAL_FEATURES), where_small)
        frames = {
            "A_new": period_slices(a_frame, "val2"),
            "manual": period_slices(m_frame, "val2"),
            "W_auto": period_slices(w_frame, "val2"),
        }
        run_fixed_block(state, "val2", frames, VAL2_LAST, "train")
        state["done"].append("diffs|val2")
        save_state(state)
        del w_frame, a_frame, m_frame, frames
        gc.collect()

    apply_decision(state)
    run_c(state, "val1", C_ROUNDS)
    run_c(state, "val2", C_ROUNDS)
    print(f"W_auto деревьев {rounds_of(state, 'W_auto')}", flush=True)
    w_test_rounds = rounds_of(state, "W_test") if any(
        row["name"] == "W_test" and row["protocol"] == "early_iterations" for row in state["rounds"]
    ) else rounds_of(state, "W_auto")
    print(f"W_test деревьев {w_test_rounds}", flush=True)
    print(f"записано {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
