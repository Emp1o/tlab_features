"""ПОСТ-ХОК, ОПИСАТЕЛЬНО. Это не предрегистрированный прогон и не критерий.

Один запуск после заморозки автопоиска. Оценивает на test:
C (152 признака, 78 деревьев), manual (27, 89) и W_auto.
W_auto здесь — список, отобранный на train по 2015-08-20, и его медиана
best_iteration на val1 (92 дерева). Список train2 (29 признаков, 90 деревьев)
в это обучение не подставляется: по правилу val2 он A_new не заменил.

A_new учится тем же замороженным протоколом (21 признак, 80 деревьев, сиды
42–46, колонки по алфавиту, train по 2015-08-20) только чтобы сверить
групповые метрики с report/test_run.csv. Если NDCG@3, NDCG@5, MRR или AUC
расходятся больше чем на 1e-12, скрипт останавливается и файл не пишет.
Опубликованные числа A_new не заменяются. Popularity пересчитывается тем же
запросом и сверяется так же.

Бутстреп против A_new и против popularity парный и описательный. Он не меняет
вывод основного test.

Предохранители те же, что у scripts/run_test.py: sha256 из FREEZE, отдельная
таблица «sha256 файлов автопоиска», окно 2015-09-04..2015-09-17 и 206150 строк,
ключ группы (user_id, day), отказ перезаписывать report/test_posthoc.csv и
запрет писать в report/test_run.csv.
"""

import gc
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_test  # noqa: E402
from feature_generator import SEARCH_WINDOWS, candidate_feature_names  # noqa: E402
from features_manual import MANUAL_FEATURES  # noqa: E402
from second_period import A_NEW, SEEDS, metric_row  # noqa: E402
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

MANUAL_PATH = ROOT / "data" / "features_manual.parquet"
WSEARCH_PATH = ROOT / "data" / "feature_candidates_wsearch.parquet"
AUDIT_PATH = ROOT / "report" / "audit_baselines.csv"
WINDOW_PATH = ROOT / "report" / "window_search.csv"
SELECTION_TRAIN = ROOT / "report" / "window_selection_train.csv"
SELECTION_TRAIN2 = ROOT / "report" / "window_selection_train2.csv"
PUBLISHED_PATH = ROOT / "report" / "test_run.csv"
OUT_PATH = ROOT / "report" / "test_posthoc.csv"
STATUS_PATH = ROOT / "STATUS.md"
TRAIN_LAST = run_test.TRAIN_LAST
TRAIN_ROWS = 1848927
A_NEW_ROUNDS = 80
C_ROUNDS = 78
MANUAL_ROUNDS = 89
W_TEST_ROUNDS = 92
AUTO_HASH_MARKER = "sha256 файлов автопоиска:"
AUTO_FILES = (
    "data/feature_candidates_wsearch.parquet",
    "report/window_search.csv",
    "report/window_selection_train.csv",
    "report/window_selection_train2.csv",
)
# Отбор на train по 2015-08-20. Порядок — сила, как в window_selection_train.csv.
# Обучение сортирует имена по алфавиту.
W_TEST_FEATURES = (
    "user_item__addtocart__1__count",
    "user_item__addtocart__3__count",
    "user_item__addtocart__7__count",
    "user_item__addtocart__30__count",
    "user_item__view__1__count",
    "user_item__view__3__count",
    "user_item__view__7__count",
    "item__addtocart__14__count",
    "item__addtocart__3__count",
    "item__addtocart__30__count",
    "item__addtocart__7__count",
    "item__addtocart__1__count",
    "item__view__14__count",
    "item__view__3__count",
    "item__view__5__share",
    "item__view__1__count",
    "item__transaction__21__count",
    "item__transaction__10__count",
    "item__view__all__recency",
    "user_item__addtocart__7_30__trend",
    "item__transaction__3__count",
    "item__transaction__5__count",
    "item__transaction__7_30__trend",
    "item__available__asof",
    "item__addtocart__all__recency",
    "user_item__addtocart__all__recency",
    "item__transaction__1__share",
    "user_item__view__7_30__trend",
)
BANNER = (
    "POST-HOC DESCRIPTIVE. Не предрегистрированный критерий и не замена "
    "report/test_run.csv. A_new пересчитан тем же протоколом только для сверки."
)
MATCH_COLUMNS = ("ndcg_3", "ndcg_5", "mrr", "auc")


def git_sha(*args):
    try:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError("не удалось прочитать git sha") from exc


def auto_file_hashes():
    text = STATUS_PATH.read_text(encoding="utf-8")
    start = text.find(AUTO_HASH_MARKER)
    if start < 0:
        raise RuntimeError("в STATUS.md нет таблицы sha256 файлов автопоиска")
    hashes = {}
    for line in text[start:].splitlines():
        if line.startswith("## "):
            break
        matched = run_test.HASH_LINE.match(line.strip())
        if matched:
            hashes[matched.group(1)] = matched.group(2)
    if set(hashes) != set(AUTO_FILES):
        raise RuntimeError("таблица sha256 файлов автопоиска не совпала с ожидаемым списком")
    actual = {}
    for rel in AUTO_FILES:
        path = ROOT / rel
        if not path.is_file():
            raise RuntimeError(f"нет входного файла {rel}")
        found = run_test.file_sha256(path)
        if found != hashes[rel]:
            raise RuntimeError(f"sha256 не совпал с заморозкой автопоиска: {rel}")
        actual[rel] = found
        print(f"sha256 ok {rel}", flush=True)
    return actual


def audit_rounds(name):
    if not AUDIT_PATH.is_file():
        raise RuntimeError("нет report/audit_baselines.csv")
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


def assert_lists_frozen():
    if len(W_TEST_FEATURES) != 28 or len(set(W_TEST_FEATURES)) != 28:
        raise RuntimeError("замороженный список W_test — не 28 разных признаков")
    pool = set(candidate_feature_names(SEARCH_WINDOWS))
    if len(pool) != 272 or not set(W_TEST_FEATURES) <= pool:
        raise RuntimeError("список W_test не входит в сетку поиска из 272 признаков")
    window = pd.read_csv(WINDOW_PATH)
    listed = window[
        (window["record"] == "list")
        & (window["period"] == "train")
        & (window["name"] == "W_test")
    ]["feature"].dropna().tolist()
    if listed != list(W_TEST_FEATURES):
        raise RuntimeError("список W_test в window_search.csv не совпал с кодом")
    rounds = window[
        (window["record"] == "rounds")
        & (window["name"] == "W_test")
        & (window["period"] == "val1")
        & (window["protocol"] == "early_iterations")
    ]
    if len(rounds) != 1 or int(rounds.iloc[0]["rounds"]) != W_TEST_ROUNDS:
        raise RuntimeError("медиана W_test в window_search.csv не равна 92")
    selected = pd.read_csv(SELECTION_TRAIN)
    passed = selected.loc[selected["прошёл"] == 1, "признак"].tolist()
    if passed != list(W_TEST_FEATURES):
        raise RuntimeError("отбор на train по 2015-08-20 не совпал со списком W_test")
    selected2 = pd.read_csv(SELECTION_TRAIN2)
    n_train2 = int((selected2["прошёл"] == 1).sum())
    if n_train2 != 29:
        raise RuntimeError(f"отбор train2: {n_train2} признаков, ожидалось 29")
    if audit_rounds("C") != C_ROUNDS or audit_rounds("manual") != MANUAL_ROUNDS:
        raise RuntimeError("медианы C или manual в аудите не 78 и 89")
    if len(A_NEW) != 21 or len(MANUAL_FEATURES) != 27:
        raise RuntimeError("A_new или manual сменили длину")


def load_frame(columns, where_sql, path):
    quoted = ", ".join(f'f."{name}"::FLOAT AS "{name}"' for name in columns)
    con = duckdb.connect()
    frame = con.sql(f"""
        SELECT e.user_id, e.item_id, e.day, e.label, e.split, {quoted}
        FROM '{EXAMPLES_PATH}' e
        JOIN '{path}' f USING (user_id, item_id, day)
        WHERE {where_sql}
        ORDER BY e.user_id, e.day, e.item_id
    """).df()
    con.close()
    return frame


def assert_train(train):
    if len(train) != TRAIN_ROWS or not train["split"].eq("train").all():
        raise RuntimeError(f"train: строк {len(train)}, ожидалось {TRAIN_ROWS} из split=train")
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
        print(f"сид {seed}, деревьев {n_rounds}, признаков {len(columns)}", flush=True)
        params["random_state"] = seed
        model = lgb.LGBMClassifier(**params)
        model.fit(train[columns], train["label"].to_numpy())
        scores = model.predict_proba(test[columns], num_iteration=int(n_rounds))[:, 1]
        grouped = per_group_metrics(test, scores, seed=TIE_SEED)
        seed_ndcg.append(float(grouped["ndcg3"].mean()))
        total = scores if total is None else total + scores
        del model, grouped
    return total / len(SEEDS), seed_ndcg


def assert_matches(fresh, stored, name):
    for column in MATCH_COLUMNS:
        gap = abs(float(fresh[column]) - float(stored[column]))
        if gap > 1e-12:
            raise RuntimeError(
                f"пересчитанный {name} разошёлся с test_run.csv по {column}: {gap}"
            )
    if int(fresh["n_groups"]) != int(stored["n_groups"]):
        raise RuntimeError(f"число групп {name} не совпало с test_run.csv")


def keep_groups(grouped):
    return {
        "ndcg3": grouped["ndcg3"],
        "user_id": grouped["user_id"],
        "day": grouped["day"],
    }


def diff_row(name, left, right, indices):
    if not same_groups(left, right):
        raise RuntimeError(f"{name}: группы разошлись по (user_id, day)")
    estimate, low, high = diff_ci(left["ndcg3"], right["ndcg3"], indices)
    return {
        "record": "diff",
        "period": "test",
        "name": name,
        "n_groups": len(left["ndcg3"]),
        "ndcg_3": estimate,
        "ndcg_3_low": low,
        "ndcg_3_high": high,
        "note": "парный бутстреп 1000, seed 42; описательно, не предрегистрированный критерий",
    }


def seed_row(name, values, n_rounds):
    return {
        "record": "seeds",
        "period": "test",
        "name": name,
        "ndcg_3_min": min(values),
        "ndcg_3_max": max(values),
        "ndcg_3_seeds": ",".join(f"{value:.6f}" for value in values),
        "iterations": ",".join(str(n_rounds) for _ in SEEDS),
        "note": BANNER,
    }


def fit_split(name, features, n_rounds, path):
    train_sql = f"e.split = 'train' AND e.day <= DATE '{TRAIN_LAST}'"
    train = load_frame(features, train_sql, path)
    test = load_frame(features, "e.split = 'test'", path)
    assert_train(train)
    run_test.assert_test_window(test)
    print(f"{name} train {len(train)}  test {len(test)}", flush=True)
    scores, seeds = fit_frozen(train, test, features, n_rounds)
    row, grouped = metric_row(
        "test", name, test, scores, TIE_SEED, len(features), [n_rounds] * len(SEEDS)
    )
    row["note"] = BANNER
    row["rounds"] = int(n_rounds)
    del train, scores
    gc.collect()
    return row, keep_groups(grouped), seeds, test


def main():
    print(BANNER, flush=True)
    if OUT_PATH.exists():
        raise RuntimeError(f"уже есть {OUT_PATH}: повторный пост-хок запрещён")
    if OUT_PATH.resolve() == PUBLISHED_PATH.resolve():
        raise RuntimeError("пост-хок не пишет в report/test_run.csv")
    original = run_test.verify_freeze_hashes()
    extra = auto_file_hashes()
    head = git_sha("rev-parse", "HEAD")
    src_sha = git_sha("log", "-1", "--format=%H", "--", "src", "scripts")
    assert_lists_frozen()
    names = candidate_names()
    if len(names) != 152:
        raise RuntimeError(f"в пуле C {len(names)} признаков, ожидалось 152")
    published = published_metrics()
    published_stamp = PUBLISHED_PATH.stat().st_mtime_ns
    print(
        f"деревья A_new {A_NEW_ROUNDS}, C {C_ROUNDS}, manual {MANUAL_ROUNDS}, "
        f"W_auto {W_TEST_ROUNDS}; sha {head}; код {src_sha}",
        flush=True,
    )

    a_row, a_grouped, a_seeds, a_test = fit_split(
        "A_new", list(A_NEW), A_NEW_ROUNDS, CANDIDATES_PATH
    )
    assert_matches(a_row, published["A_new"], "A_new")
    popularity = run_test.popularity_scores(a_test)
    pop_row, pop_grouped_full = metric_row(
        "test", "popularity", a_test, popularity, 0, None, []
    )
    pop_row["logloss"] = float("nan")
    pop_row["note"] = BANNER
    assert_matches(pop_row, published["popularity"], "popularity")
    pop_grouped = keep_groups(pop_grouped_full)
    del a_test, popularity, pop_grouped_full
    gc.collect()

    c_row, c_grouped, c_seeds, c_test = fit_split("C", names, C_ROUNDS, CANDIDATES_PATH)
    del c_test
    m_row, m_grouped, m_seeds, m_test = fit_split(
        "manual", list(MANUAL_FEATURES), MANUAL_ROUNDS, MANUAL_PATH
    )
    del m_test
    w_row, w_grouped, w_seeds, w_test = fit_split(
        "W_auto", list(W_TEST_FEATURES), W_TEST_ROUNDS, WSEARCH_PATH
    )
    del w_test
    gc.collect()

    for label, grouped in (
        ("C", c_grouped),
        ("manual", m_grouped),
        ("W_auto", w_grouped),
        ("popularity", pop_grouped),
    ):
        if not same_groups(a_grouped, grouped):
            raise RuntimeError(f"группы A_new и {label} разошлись по (user_id, day)")

    indices = bootstrap_indices(len(a_grouped["ndcg3"]), N_BOOTSTRAP, BOOTSTRAP_SEED)
    rows = [{
        "record": "banner",
        "period": "test",
        "name": "posthoc",
        "note": BANNER,
    }, {
        "record": "freeze",
        "period": "test",
        "name": "code_sha",
        "note": head,
    }, {
        "record": "freeze",
        "period": "test",
        "name": "src_scripts_sha",
        "note": src_sha,
    }, {
        "record": "freeze",
        "period": "test",
        "name": "rounds_A_new",
        "n_features": len(A_NEW),
        "rounds": A_NEW_ROUNDS,
        "note": "замороженные 80; пересчёт только для сверки с report/test_run.csv, допуск 1e-12",
    }, {
        "record": "freeze",
        "period": "test",
        "name": "rounds_C",
        "n_features": len(names),
        "rounds": C_ROUNDS,
        "note": "медиана best_iteration C на val1, сиды 42–46; заново не выбиралась",
    }, {
        "record": "freeze",
        "period": "test",
        "name": "rounds_manual",
        "n_features": len(MANUAL_FEATURES),
        "rounds": MANUAL_ROUNDS,
        "note": "медиана best_iteration manual на val1, сиды 42–46; заново не выбиралась",
    }, {
        "record": "freeze",
        "period": "test",
        "name": "rounds_W_auto",
        "n_features": len(W_TEST_FEATURES),
        "rounds": W_TEST_ROUNDS,
        "note": (
            "на test это список отбора на train по 2015-08-20 и медиана best_iteration "
            "этого списка на val1; список train2 (29 признаков, 90 деревьев) сюда не входит"
        ),
    }, {
        "record": "freeze",
        "period": "test",
        "name": "W_auto_features",
        "n_features": len(W_TEST_FEATURES),
        "note": ",".join(W_TEST_FEATURES),
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
    for rel, digest in {**original, **extra}.items():
        rows.append({
            "record": "freeze",
            "period": "test",
            "name": "data_sha256",
            "note": f"{rel} {digest}",
        })
    rows.extend([
        a_row,
        pop_row,
        c_row,
        m_row,
        w_row,
        diff_row("C-A_new", c_grouped, a_grouped, indices),
        diff_row("manual-A_new", m_grouped, a_grouped, indices),
        diff_row("W_auto-A_new", w_grouped, a_grouped, indices),
        diff_row("C-popularity", c_grouped, pop_grouped, indices),
        diff_row("manual-popularity", m_grouped, pop_grouped, indices),
        diff_row("W_auto-popularity", w_grouped, pop_grouped, indices),
        seed_row("A_new", a_seeds, A_NEW_ROUNDS),
        seed_row("C", c_seeds, C_ROUNDS),
        seed_row("manual", m_seeds, MANUAL_ROUNDS),
        seed_row("W_auto", w_seeds, W_TEST_ROUNDS),
    ])
    if OUT_PATH.exists():
        raise RuntimeError(f"уже есть {OUT_PATH}: повторный пост-хок запрещён")
    if PUBLISHED_PATH.stat().st_mtime_ns != published_stamp:
        raise RuntimeError("report/test_run.csv изменился во время пост-хока")
    report = pd.DataFrame(rows)
    with OUT_PATH.open("x", encoding="utf-8") as handle:
        report.to_csv(handle, index=False)
    print(report.to_string(index=False))
    print(f"записано {OUT_PATH}")


if __name__ == "__main__":
    main()
