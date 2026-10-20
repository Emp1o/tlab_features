"""Один прогон на test. Предрегистрация — в STATUS.md, раздел «Предрегистрация теста (Максим, 20.10.2026)».

Прогон уже выполнен один раз 20.10.2026. Повтор запрещён: скрипт не перезаписывает
report/test_run.csv. Результат — в STATUS.md, раздел «Результат test», и в этом файле.

Обучение — строки split='train' с day <= 2015-08-20. Признаки — замороженный
A_NEW, колонки по алфавиту. Деревьев ровно FIXED_ROUNDS (медиана best_iteration
A_new на val1). Метки test и val в fit не входят. Сиды 42–46, метрика модели —
среднее вероятностей.

Оценка — split='test', окно 2015-09-04..2015-09-17, 206150 строк. Группа
ранжирования — (user_id, day). NDCG@5, MRR и AUC только описывают прогон.
Критерий один: нижняя граница 95% CI NDCG@3 (A_new − popularity) > 0.
Бутстреп парный, 1000 повторов, seed 42.

Перед обучением сверяются sha256 всех входных файлов из раздела FREEZE в
STATUS.md. Повторный запуск не перезаписывает report/test_run.csv.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

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
    same_groups,
)

EVENTS_PATH = ROOT / "data" / "events_clean.parquet"
STATUS_PATH = ROOT / "STATUS.md"
OUT_PATH = ROOT / "report" / "test_run.csv"
TRAIN_LAST = "2015-08-20"
TEST_FIRST = "2015-09-04"
TEST_LAST = "2015-09-17"
TEST_ROWS = 206150
# Медиана best_iteration A_new на val1 по сидам 42–46: 77, 80, 91, 84, 78.
FIXED_ROUNDS = 80
HASH_LINE = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*`([0-9a-f]{64})`\s*\|$")


def freeze_file_hashes():
    """Пары путь → sha256 из таблицы FREEZE в STATUS.md."""
    text = STATUS_PATH.read_text(encoding="utf-8")
    marker = "sha256 входных файлов:"
    start = text.find(marker)
    if start < 0:
        raise RuntimeError("в STATUS.md нет таблицы sha256")
    hashes = {}
    for line in text[start:].splitlines():
        if line.startswith("## "):
            break
        matched = HASH_LINE.match(line.strip())
        if matched:
            hashes[matched.group(1)] = matched.group(2)
    if not hashes:
        raise RuntimeError("таблица sha256 в STATUS.md пуста")
    return hashes


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_freeze_hashes():
    expected = freeze_file_hashes()
    actual = {}
    for rel, wanted in expected.items():
        path = ROOT / rel
        if not path.is_file():
            raise RuntimeError(f"нет входного файла {rel}")
        found = file_sha256(path)
        if found != wanted:
            raise RuntimeError(f"sha256 не совпал с FREEZE: {rel}")
        actual[rel] = found
        print(f"sha256 ok {rel}", flush=True)
    return actual


def code_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("не удалось прочитать git sha кода") from exc


def frozen_params():
    params = dict(PARAMS)
    params["n_estimators"] = int(FIXED_ROUNDS)
    return params


def load_split(split_sql):
    columns = sorted(A_NEW)
    quoted = ", ".join(f'c."{name}"::FLOAT AS "{name}"' for name in columns)
    con = duckdb.connect()
    return con.sql(f"""
        SELECT e.user_id, e.item_id, e.day, e.label, e.split, {quoted}
        FROM '{EXAMPLES_PATH}' e
        JOIN '{CANDIDATES_PATH}' c USING (user_id, item_id, day)
        WHERE {split_sql}
        ORDER BY e.user_id, e.day, e.item_id
    """).df()


def assert_test_window(test):
    if not test["split"].eq("test").all():
        raise RuntimeError("в оценке есть строки не из test")
    days = pd.to_datetime(test["day"])
    first = pd.Timestamp(TEST_FIRST)
    last = pd.Timestamp(TEST_LAST)
    if len(test) != TEST_ROWS or days.min() != first or days.max() != last:
        raise RuntimeError(
            f"окно test: строк {len(test)}, дни {days.min().date()}..{days.max().date()}; "
            f"ожидалось {TEST_ROWS} строк, {TEST_FIRST}..{TEST_LAST}"
        )
    if ((days < first) | (days > last)).any():
        raise RuntimeError("в test есть дни вне 2015-09-04..2015-09-17")


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
    columns = sorted(features)
    params_base = frozen_params()
    total = None
    seed_ndcg = []
    for seed in SEEDS:
        print(f"сид {seed}", flush=True)
        params = dict(params_base)
        params["random_state"] = seed
        model = lgb.LGBMClassifier(**params)
        model.fit(train[columns], train["label"].to_numpy())
        scores = model.predict_proba(test[columns], num_iteration=int(FIXED_ROUNDS))[:, 1]
        grouped = per_group_metrics(test, scores, seed=TIE_SEED)
        seed_ndcg.append(float(grouped["ndcg3"].mean()))
        total = scores if total is None else total + scores
    return total / len(SEEDS), seed_ndcg


def provenance_rows(sha, hashes):
    rows = [{
        "record": "freeze",
        "period": "test",
        "name": "code_sha",
        "note": sha,
    }, {
        "record": "freeze",
        "period": "test",
        "name": "rounds",
        "n_features": FIXED_ROUNDS,
        "note": "медиана best_iteration A_new на val1",
    }, {
        "record": "freeze",
        "period": "test",
        "name": "seeds",
        "note": ",".join(str(seed) for seed in SEEDS),
    }, {
        "record": "freeze",
        "period": "test",
        "name": "params",
        "note": json.dumps(frozen_params(), sort_keys=True, ensure_ascii=True),
    }, {
        "record": "freeze",
        "period": "test",
        "name": "group_key",
        "note": "user_id,day",
    }, {
        "record": "freeze",
        "period": "test",
        "name": "criterion",
        "note": "NDCG@3 (A_new-popularity) 95% CI lower bound > 0; NDCG@5, MRR, AUC descriptive",
    }]
    for rel, digest in hashes.items():
        rows.append({
            "record": "freeze",
            "period": "test",
            "name": "data_sha256",
            "note": f"{rel} {digest}",
        })
    return rows


def main():
    if OUT_PATH.exists():
        raise RuntimeError(f"уже есть {OUT_PATH}: повторный прогон запрещён")
    hashes = verify_freeze_hashes()
    sha = code_sha()
    train = load_split(f"e.split = 'train' AND e.day <= DATE '{TRAIN_LAST}'")
    test = load_split("e.split = 'test'")
    if len(train) == 0 or len(test) == 0:
        raise RuntimeError("пустое обучение или test")
    if not train["split"].eq("train").all() or pd.to_datetime(train["day"]).max() > pd.Timestamp(TRAIN_LAST):
        raise RuntimeError("в обучении есть дни после 2015-08-20")
    assert_test_window(test)
    print(f"train {len(train)}  test {len(test)}  деревьев {FIXED_ROUNDS}  sha {sha}", flush=True)
    scores, seed_ndcg = fit_scores(train, test, A_NEW)
    row, grouped = metric_row("test", "A_new", test, scores, TIE_SEED, len(A_NEW), [FIXED_ROUNDS] * len(SEEDS))
    popularity = popularity_scores(test)
    pop_row, pop_grouped = metric_row("test", "popularity", test, popularity, 0, None, [])
    if not same_groups(grouped, pop_grouped):
        raise RuntimeError("группы A_new и popularity разошлись по (user_id, day)")
    indices = bootstrap_indices(len(grouped["ndcg3"]), N_BOOTSTRAP, BOOTSTRAP_SEED)
    estimate, low, high = diff_ci(grouped["ndcg3"], pop_grouped["ndcg3"], indices)
    criterion_met = bool(low > 0)
    rows = provenance_rows(sha, hashes)
    rows.extend([
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
            "note": "paired bootstrap 1000 seed 42; criterion is ndcg_3_low > 0",
        },
        {
            "record": "criterion",
            "period": "test",
            "name": "A_new-popularity",
            "n_groups": len(grouped["ndcg3"]),
            "ndcg_3": estimate,
            "ndcg_3_low": low,
            "ndcg_3_high": high,
            "note": "pass" if criterion_met else "fail",
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
    ])
    for item in rows:
        item.setdefault("record", "metrics")
    report = pd.DataFrame(rows)
    if OUT_PATH.exists():
        raise RuntimeError(f"уже есть {OUT_PATH}: повторный прогон запрещён")
    with OUT_PATH.open("x", encoding="utf-8") as handle:
        report.to_csv(handle, index=False)
    print(report.to_string(index=False))
    print(f"A_new − popularity: {estimate:.6f} ({low:.6f}–{high:.6f})")
    print(f"критерий lower bound > 0: {'pass' if criterion_met else 'fail'}")
    print(f"записано {OUT_PATH}")


if __name__ == "__main__":
    main()
