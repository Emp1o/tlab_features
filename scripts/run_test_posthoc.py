"""ПОСТ-ХОК, ОПИСАТЕЛЬНО. Это не предрегистрированный прогон и не критерий.

Скрипт оценивает замороженные C (все 152 признака) и 27 ручных признаков
на test. A_new заново не обучается: его числа и числа popularity берутся из
report/test_run.csv. Популярность пересчитывается тем же запросом, что в
scripts/run_test.py, и прогон прерывается, если NDCG@3 не совпал с уже
записанным. Парный бутстреп против A_new не считается: групповые предсказания
A_new в test_run.csv не сохранены.

Число деревьев — медиана best_iteration на val1 из report/audit_baselines.csv
(сиды 42–46). На test eval_set нет. Результат пишется только в
report/test_posthoc.csv и не заменяет report/test_run.csv.

Те же предохранители, что у scripts/run_test.py: sha256 из FREEZE, окно
2015-09-04..2015-09-17 и 206150 строк, ключ группы (user_id, day), отказ
перезаписывать выходной файл.

Не запускать вместе с предрегистрированным прогоном. Один запуск, когда
отчёт аудита уже записан.
"""

import json
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_test  # noqa: E402
from features_manual import MANUAL_FEATURES  # noqa: E402
from second_period import SEEDS, metric_row  # noqa: E402
from train_lgbm import (  # noqa: E402
    BOOTSTRAP_SEED,
    CANDIDATES_PATH,
    EXAMPLES_PATH,
    N_BOOTSTRAP,
    PARAMS,
    TIE_SEED,
    bootstrap_indices,
    candidate_names,
    diff_ci,
    per_group_metrics,
    same_groups,
)

# train_lgbm не держит путь ручных признаков.
MANUAL_PATH = ROOT / "data" / "features_manual.parquet"
AUDIT_PATH = ROOT / "report" / "audit_baselines.csv"
PUBLISHED_PATH = ROOT / "report" / "test_run.csv"
OUT_PATH = ROOT / "report" / "test_posthoc.csv"
TRAIN_LAST = run_test.TRAIN_LAST
BANNER = (
    "POST-HOC DESCRIPTIVE. Не предрегистрированный критерий и не замена "
    "report/test_run.csv. A_new не переобучался."
)


def frozen_rounds(name):
    if not AUDIT_PATH.is_file():
        raise RuntimeError("нет report/audit_baselines.csv: медиану деревьев брать неоткуда")
    table = pd.read_csv(AUDIT_PATH)
    rows = table[
        (table["record"] == "rounds")
        & (table["name"] == name)
        & (table["period"] == "val1")
        & (table["protocol"] == "early_iterations")
    ]
    if len(rows) != 1 or pd.isna(rows.iloc[0]["rounds"]):
        raise RuntimeError(f"в аудите нет одной медианы val1 для {name}")
    return int(rows.iloc[0]["rounds"])


def published_metrics():
    if not PUBLISHED_PATH.is_file():
        raise RuntimeError("нет report/test_run.csv")
    table = pd.read_csv(PUBLISHED_PATH)
    picked = {}
    for name in ("A_new", "popularity"):
        rows = table[(table["record"] == "metrics") & (table["name"] == name)]
        if len(rows) != 1:
            raise RuntimeError(f"в test_run.csv нет одной строки metrics/{name}")
        picked[name] = rows.iloc[0]
    return picked


def load_frame(columns, where_sql, source):
    if source == "candidates":
        path = CANDIDATES_PATH
    elif source == "manual":
        path = MANUAL_PATH
    else:
        raise RuntimeError(f"неизвестный источник {source}")
    quoted = ", ".join(f'f."{name}"::FLOAT AS "{name}"' for name in columns)
    con = duckdb.connect()
    frame = con.sql(f"""
        SELECT e.user_id, e.item_id, e.day, e.label, e.split, {quoted}
        FROM '{EXAMPLES_PATH}' e
        JOIN '{path}' f USING (user_id, item_id, day)
        WHERE {where_sql}
        ORDER BY e.user_id, e.day, e.item_id
    """).df()
    return frame


def assert_train(train):
    if len(train) == 0 or not train["split"].eq("train").all():
        raise RuntimeError("в обучении есть строки не из train")
    if pd.to_datetime(train["day"]).max() > pd.Timestamp(TRAIN_LAST):
        raise RuntimeError("в обучении есть дни после 2015-08-20")


def fit_frozen(train, test, features, n_rounds):
    """Ровно n_rounds деревьев. Метки test в fit не входят."""
    columns = sorted(features)
    params = dict(PARAMS)
    params["n_estimators"] = int(n_rounds)
    total = None
    seed_ndcg = []
    for seed in SEEDS:
        print(f"сид {seed}, деревьев {n_rounds}", flush=True)
        params["random_state"] = seed
        model = lgb.LGBMClassifier(**params)
        model.fit(train[columns], train["label"].to_numpy())
        scores = model.predict_proba(test[columns], num_iteration=int(n_rounds))[:, 1]
        grouped = per_group_metrics(test, scores, seed=TIE_SEED)
        seed_ndcg.append(float(grouped["ndcg3"].mean()))
        total = scores if total is None else total + scores
        del model
    return total / len(SEEDS), seed_ndcg


def copied_metric(stored, name):
    row = {
        "record": "metrics",
        "period": "test",
        "name": name,
        "n_features": stored["n_features"],
        "n_groups": stored["n_groups"],
        "ndcg_3": stored["ndcg_3"],
        "ndcg_5": stored["ndcg_5"],
        "mrr": stored["mrr"],
        "auc": stored["auc"],
        "iterations": stored["iterations"],
        "note": "скопировано из report/test_run.csv; модель не переобучалась. Recall и logloss в том прогоне не считались.",
    }
    return row


def assert_popularity_matches(fresh, stored):
    for column in ("ndcg_3", "ndcg_5", "mrr", "auc"):
        gap = abs(float(fresh[column]) - float(stored[column]))
        if gap > 1e-12:
            raise RuntimeError(
                f"пересчитанная popularity разошлась с test_run.csv по {column}: {gap}"
            )
    if int(fresh["n_groups"]) != int(stored["n_groups"]):
        raise RuntimeError("число групп popularity не совпало с test_run.csv")


def mean_gap_row(name, left, right):
    return {
        "record": "diff_of_means",
        "period": "test",
        "name": name,
        "n_groups": left["n_groups"],
        "ndcg_3": float(left["ndcg_3"]) - float(right["ndcg_3"]),
        "note": (
            "разность средних NDCG@3, не парный бутстреп: групповые предсказания "
            "A_new в report/test_run.csv не сохранены, A_new не переобучался"
        ),
    }


def main():
    print(BANNER, flush=True)
    if OUT_PATH.exists():
        raise RuntimeError(f"уже есть {OUT_PATH}: повторный пост-хок запрещён")
    if OUT_PATH.resolve() == PUBLISHED_PATH.resolve():
        raise RuntimeError("пост-хок не пишет в report/test_run.csv")
    hashes = run_test.verify_freeze_hashes()
    sha = run_test.code_sha()
    published = published_metrics()
    manual_rounds = frozen_rounds("manual")
    c_rounds = frozen_rounds("C")
    print(f"деревья manual {manual_rounds}, C {c_rounds}, sha {sha}", flush=True)

    names = candidate_names()
    if len(names) != 152:
        raise RuntimeError(f"в пуле {len(names)} признаков, ожидалось 152")
    if len(MANUAL_FEATURES) != 27:
        raise RuntimeError("ручных признаков не 27")

    train_sql = f"e.split = 'train' AND e.day <= DATE '{TRAIN_LAST}'"
    c_train = load_frame(names, train_sql, "candidates")
    c_test = load_frame(names, "e.split = 'test'", "candidates")
    assert_train(c_train)
    run_test.assert_test_window(c_test)
    print(f"C train {len(c_train)}  test {len(c_test)}", flush=True)
    c_scores, c_seeds = fit_frozen(c_train, c_test, names, c_rounds)
    c_row, c_grouped = metric_row(
        "test", "C", c_test, c_scores, TIE_SEED, len(names), [c_rounds] * len(SEEDS)
    )
    n_c_test = len(c_test)
    del c_train, c_scores, c_test

    m_train = load_frame(list(MANUAL_FEATURES), train_sql, "manual")
    m_test = load_frame(list(MANUAL_FEATURES), "e.split = 'test'", "manual")
    assert_train(m_train)
    run_test.assert_test_window(m_test)
    if len(m_test) != n_c_test:
        raise RuntimeError("число строк test у manual и C разошлось")
    print(f"manual train {len(m_train)}  test {len(m_test)}", flush=True)
    m_scores, m_seeds = fit_frozen(m_train, m_test, MANUAL_FEATURES, manual_rounds)
    m_row, m_grouped = metric_row(
        "test", "manual", m_test, m_scores, TIE_SEED, len(MANUAL_FEATURES), [manual_rounds] * len(SEEDS)
    )
    del m_train, m_scores

    popularity = run_test.popularity_scores(m_test)
    pop_row, pop_grouped = metric_row("test", "popularity_recomputed", m_test, popularity, 0, None, [])
    pop_row["logloss"] = float("nan")
    assert_popularity_matches(pop_row, published["popularity"])
    if not same_groups(c_grouped, pop_grouped) or not same_groups(m_grouped, pop_grouped):
        raise RuntimeError("группы C, manual и popularity разошлись по (user_id, day)")
    if not same_groups(c_grouped, m_grouped):
        raise RuntimeError("группы C и manual разошлись по (user_id, day)")

    indices = bootstrap_indices(len(pop_grouped["ndcg3"]), N_BOOTSTRAP, BOOTSTRAP_SEED)
    c_est, c_low, c_high = diff_ci(c_grouped["ndcg3"], pop_grouped["ndcg3"], indices)
    m_est, m_low, m_high = diff_ci(m_grouped["ndcg3"], pop_grouped["ndcg3"], indices)

    rows = [{
        "record": "banner",
        "period": "test",
        "name": "posthoc",
        "note": BANNER,
    }, {
        "record": "freeze",
        "period": "test",
        "name": "code_sha",
        "note": sha,
    }, {
        "record": "freeze",
        "period": "test",
        "name": "rounds_manual",
        "n_features": len(MANUAL_FEATURES),
        "rounds": manual_rounds,
        "note": "медиана best_iteration manual на val1, сиды 42–46, из report/audit_baselines.csv",
    }, {
        "record": "freeze",
        "period": "test",
        "name": "rounds_C",
        "n_features": len(names),
        "rounds": c_rounds,
        "note": "медиана best_iteration C на val1, сиды 42–46, из report/audit_baselines.csv",
    }, {
        "record": "freeze",
        "period": "test",
        "name": "seeds",
        "note": ",".join(str(seed) for seed in SEEDS),
    }, {
        "record": "freeze",
        "period": "test",
        "name": "params",
        "note": json.dumps(dict(PARAMS), sort_keys=True, ensure_ascii=True),
    }, {
        "record": "freeze",
        "period": "test",
        "name": "group_key",
        "note": "user_id,day",
    }, {
        "record": "freeze",
        "period": "test",
        "name": "not_criterion",
        "note": "описательные разницы; предрегистрированный критерий A_new−popularity уже в report/test_run.csv",
    }]
    for rel, digest in hashes.items():
        rows.append({
            "record": "freeze",
            "period": "test",
            "name": "data_sha256",
            "note": f"{rel} {digest}",
        })
    a_copy = copied_metric(published["A_new"], "A_new")
    p_copy = copied_metric(published["popularity"], "popularity")
    for item in (c_row, m_row, pop_row):
        item["note"] = BANNER
    rows.extend([
        a_copy,
        p_copy,
        c_row,
        m_row,
        pop_row,
        {
            "record": "diff",
            "period": "test",
            "name": "C-popularity",
            "n_groups": len(c_grouped["ndcg3"]),
            "ndcg_3": c_est,
            "ndcg_3_low": c_low,
            "ndcg_3_high": c_high,
            "note": "парный бутстреп 1000, seed 42; описательно, не предрегистрированный критерий",
        },
        {
            "record": "diff",
            "period": "test",
            "name": "manual-popularity",
            "n_groups": len(m_grouped["ndcg3"]),
            "ndcg_3": m_est,
            "ndcg_3_low": m_low,
            "ndcg_3_high": m_high,
            "note": "парный бутстреп 1000, seed 42; описательно, не предрегистрированный критерий",
        },
        mean_gap_row("C-A_new", c_row, a_copy),
        mean_gap_row("manual-A_new", m_row, a_copy),
        {
            "record": "seeds",
            "period": "test",
            "name": "C",
            "ndcg_3_min": min(c_seeds),
            "ndcg_3_max": max(c_seeds),
            "ndcg_3_seeds": ",".join(f"{value:.6f}" for value in c_seeds),
            "iterations": ",".join(str(c_rounds) for _ in SEEDS),
            "note": BANNER,
        },
        {
            "record": "seeds",
            "period": "test",
            "name": "manual",
            "ndcg_3_min": min(m_seeds),
            "ndcg_3_max": max(m_seeds),
            "ndcg_3_seeds": ",".join(f"{value:.6f}" for value in m_seeds),
            "iterations": ",".join(str(manual_rounds) for _ in SEEDS),
            "note": BANNER,
        },
    ])
    if OUT_PATH.exists():
        raise RuntimeError(f"уже есть {OUT_PATH}: повторный пост-хок запрещён")
    report = pd.DataFrame(rows)
    with OUT_PATH.open("x", encoding="utf-8") as handle:
        report.to_csv(handle, index=False)
    print(report.to_string(index=False))
    print(f"записано {OUT_PATH}")


if __name__ == "__main__":
    main()
