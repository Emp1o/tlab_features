import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CANDIDATES_PATH = "data/feature_candidates.parquet"
REPORT_PATH = "report/feature_selection.csv"
KEYS = ("user_id", "item_id", "day")


def impute_train(values):
    """NULL заменяется медианой train. Столбец из одних NULL становится 0."""
    if isinstance(values, np.ma.MaskedArray):
        values = values.filled(np.nan)
    values = np.asarray(values, dtype=np.float64)
    median = np.nanmedian(values)
    if np.isnan(median):
        median = 0.0
    return np.where(np.isnan(values), median, values)


def safe_auc(y, values):
    if np.unique(values).size < 2 or np.unique(y).size < 2:
        return 0.5
    return float(roc_auc_score(y, values))


def roc_auc_small(y, values):
    """AUC для короткой группы. Ничьи получают средний ранг, как в sklearn."""
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    order = np.argsort(values, kind="mergesort")
    ordered_values = values[order]
    ordered_y = y[order]
    ranks = np.empty(y.size, dtype=np.float64)
    start = 0
    while start < y.size:
        stop = start + 1
        while stop < y.size and ordered_values[stop] == ordered_values[start]:
            stop += 1
        ranks[start:stop] = (start + 1 + stop) / 2.0
        start = stop
    sum_pos = ranks[ordered_y == 1].sum()
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def group_bounds(user_id, day):
    """Границы групп (user, day). Строки уже отсортированы по user_id, day."""
    user_id = np.asarray(user_id)
    day = np.asarray(day)
    if day.dtype == object:
        day_key = np.array([item.toordinal() for item in day], dtype=np.int32)
    elif getattr(day.dtype, "kind", "") == "M":
        day_key = day.astype("datetime64[D]").astype(np.int32)
    else:
        day_key = day.astype(np.int64)
    change = np.empty(len(user_id), dtype=bool)
    change[0] = True
    change[1:] = (user_id[1:] != user_id[:-1]) | (day_key[1:] != day_key[:-1])
    starts = np.flatnonzero(change)
    ends = np.empty_like(starts)
    ends[:-1] = starts[1:]
    ends[-1] = len(user_id)
    return starts, ends


def summarize_group_auc(y, values, starts, ends):
    """Средний AUC по группам, где метки не все одинаковые.

    Группа с постоянным признаком в среднее не входит: внутри неё ранжировать
    нечего. Если так устроена каждая группа, group_auc = 0.5.
    """
    y = np.asarray(y)
    values = np.asarray(values, dtype=np.float64)
    vmin = np.minimum.reduceat(values, starts)
    vmax = np.maximum.reduceat(values, starts)
    constant_in_group = bool(np.all(vmin == vmax))
    if constant_in_group:
        return 0.5, 0, True
    mixed = np.minimum.reduceat(y, starts) != np.maximum.reduceat(y, starts)
    usable = np.flatnonzero(mixed & (vmin != vmax))
    if usable.size == 0:
        return 0.5, 0, False

    sizes = ends[usable] - starts[usable]
    aucs = np.empty(usable.size, dtype=np.float64)
    small = sizes == 2
    if np.any(small):
        left = starts[usable[small]]
        positive_first = y[left] == 1
        left_value = values[left]
        right_value = values[left + 1]
        aucs[small] = np.where(positive_first, left_value > right_value, right_value > left_value)
    for index in np.flatnonzero(~small):
        group = usable[index]
        sl = slice(int(starts[group]), int(ends[group]))
        aucs[index] = roc_auc_small(y[sl], values[sl])
    return float(aucs.mean()), int(usable.size), False


def average_ranks(values):
    """Ранги с 1. Одинаковые значения получают средний ранг."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.flatnonzero(np.r_[True, sorted_values[1:] != sorted_values[:-1]])
    counts = np.diff(np.append(starts, len(values)))
    average = starts + (counts + 1) / 2.0
    ranked = np.empty(len(values), dtype=np.float64)
    ranked[order] = np.repeat(average, counts)
    return ranked


def spearman_matrix(columns):
    """Столбцы уже без NULL, форма (n, k). У константы корреляция не определена."""
    ranks = np.column_stack([average_ranks(columns[:, j]) for j in range(columns.shape[1])])
    centered = ranks - ranks.mean(axis=0)
    std = np.sqrt((centered ** 2).sum(axis=0) / len(ranks))
    corr = np.full((columns.shape[1], columns.shape[1]), np.nan)
    good = std > 0
    if good.any():
        norm = centered[:, good] / std[good]
        block = (norm.T @ norm) / len(ranks)
        idx = np.flatnonzero(good)
        corr[np.ix_(idx, idx)] = block
    np.fill_diagonal(corr, 1.0)
    return corr


def decide(names, global_aucs, group_aucs, n_groups, constant_in_group, filled):
    """Сила — |group_auc − 0.5|. Ниже среднего выбрасываем, затем дубли по Спирмену.

    При равной силе остаётся признак, чьё имя раньше по алфавиту.
    `filled` нужен только для признаков не ниже порога.
    """
    names = list(names)
    global_aucs = np.asarray(global_aucs, dtype=np.float64)
    group_aucs = np.asarray(group_aucs, dtype=np.float64)
    n_groups = np.asarray(n_groups, dtype=np.int32)
    constant_in_group = np.asarray(constant_in_group, dtype=np.int8)
    strength = np.abs(group_aucs - 0.5)
    threshold = float(strength.mean())

    passed = np.zeros(len(names), dtype=np.int8)
    reason = [""] * len(names)
    strong = [i for i in range(len(names)) if strength[i] >= threshold]
    strong.sort(key=lambda i: (-strength[i], names[i]))
    for i in range(len(names)):
        if strength[i] < threshold:
            reason[i] = "ниже порога"

    matrix = np.column_stack([filled[names[i]] for i in strong]) if strong else np.zeros((0, 0))
    corr = spearman_matrix(matrix) if strong else np.zeros((0, 0))
    kept = []
    for local, idx in enumerate(strong):
        duplicate = None
        for kept_local in kept:
            value = corr[local, kept_local]
            if np.isfinite(value) and abs(value) > 0.9:
                duplicate = names[strong[kept_local]]
                break
        if duplicate is None:
            kept.append(local)
            passed[idx] = 1
            reason[idx] = "прошёл"
        else:
            reason[idx] = f"дубль: {duplicate}"

    table = pd.DataFrame({
        "признак": names,
        "global_auc": global_aucs,
        "group_auc": group_aucs,
        "n_groups_used": n_groups,
        "сила": strength,
        "прошёл": passed,
        "причина": reason,
        "constant_in_group": constant_in_group,
    })
    table = table.sort_values(["сила", "признак"], ascending=[False, True], ignore_index=True)
    return table, threshold


def feature_names(con, candidates_path=CANDIDATES_PATH):
    described = con.sql(f"DESCRIBE SELECT * FROM '{candidates_path}'").fetchall()
    return [row[0] for row in described if row[0] not in KEYS]


def train_filter(last_day=None):
    """split='train'. last_day отсекает дни строго после этой даты, включая val2."""
    if last_day is None:
        return "e.split = 'train'"
    return f"e.split = 'train' AND e.day <= DATE '{last_day}'"


def run_selection(last_day=None, report_path=REPORT_PATH, candidates_path=CANDIDATES_PATH):
    import duckdb

    if str(report_path) == str(REPORT_PATH) and (
        last_day is not None or str(candidates_path) != str(CANDIDATES_PATH)
    ):
        raise RuntimeError("этот отбор не перезаписывает report/feature_selection.csv")
    where = train_filter(last_day)
    con = duckdb.connect()
    # Метки val и test не читаем: порог считается только по train.
    names = feature_names(con, candidates_path)
    keys = con.sql(f"""
        SELECT user_id, day, label
        FROM 'data/examples.parquet' e
        WHERE {where}
        ORDER BY user_id, day, item_id
    """).fetchnumpy()
    y = np.asarray(keys["label"], dtype=np.int8)
    starts, ends = group_bounds(keys["user_id"], keys["day"])

    global_aucs = np.empty(len(names))
    group_aucs = np.empty(len(names))
    n_groups = np.empty(len(names), dtype=np.int32)
    constant_in_group = np.empty(len(names), dtype=np.int8)
    for start in range(0, len(names), 12):
        batch = names[start:start + 12]
        quoted = ", ".join(f'c."{name}"::DOUBLE AS "{name}"' for name in batch)
        frame = con.sql(f"""
            SELECT {quoted}
            FROM 'data/examples.parquet' e
            JOIN '{candidates_path}' c USING (user_id, item_id, day)
            WHERE {where}
            ORDER BY e.user_id, e.day, e.item_id
        """).fetchnumpy()
        if len(frame[batch[0]]) != len(y):
            raise SystemExit("train rows and feature rows diverged")
        for name in batch:
            values = impute_train(frame[name])
            index = names.index(name)
            global_aucs[index] = safe_auc(y, values)
            group_auc, used, constant = summarize_group_auc(y, values, starts, ends)
            group_aucs[index] = group_auc
            n_groups[index] = used
            constant_in_group[index] = int(constant)
        del frame
        print(f"auc: {min(start + 12, len(names))}/{len(names)}", flush=True)

    strength = np.abs(group_aucs - 0.5)
    strong_names = [name for name, value in zip(names, strength) if value >= float(strength.mean())]
    filled = {}
    if strong_names:
        quoted = ", ".join(f'c."{name}"::DOUBLE AS "{name}"' for name in strong_names)
        frame = con.sql(f"""
            SELECT {quoted}
            FROM 'data/examples.parquet' e
            JOIN '{candidates_path}' c USING (user_id, item_id, day)
            WHERE {where}
            ORDER BY e.user_id, e.day, e.item_id
        """).fetchnumpy()
        filled = {name: impute_train(frame[name]) for name in strong_names}
        del frame

    table, threshold = decide(
        names, global_aucs, group_aucs, n_groups, constant_in_group, filled
    )
    table.to_csv(report_path, index=False)
    return table, threshold


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--last-day", default=None)
    parser.add_argument("--report", default=REPORT_PATH)
    args = parser.parse_args()
    table, threshold = run_selection(args.last_day, args.report)

    n_below = int((table["причина"] == "ниже порога").sum())
    n_dup = int(table["причина"].str.startswith("дубль").sum())
    kept = table[table["прошёл"] == 1]
    print(f"кандидатов: {len(table)}")
    print(f"окно: {args.last_day or 'весь train'}")
    print(f"файл: {args.report}")
    print(f"порог: {threshold:.6f}")
    print(f"ниже порога: {n_below}")
    print(f"дублей: {n_dup}")
    print(f"прошло: {len(kept)}")
    constant_names = table.loc[table["constant_in_group"] == 1, "признак"]
    print(f"постоянные внутри группы ({len(constant_names)}):")
    for name in constant_names:
        print(f"  {name}")
    print("итоговый список:")
    show = kept[["признак", "group_auc", "global_auc", "n_groups_used", "сила"]].copy()
    show["group_auc"] = show["group_auc"].round(4)
    show["global_auc"] = show["global_auc"].round(4)
    show["сила"] = show["сила"].round(4)
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
