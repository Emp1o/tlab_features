import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def reciprocal_rank(labels):
    hits = np.flatnonzero(np.asarray(labels) == 1)
    if hits.size == 0:
        return 0.0
    return 1.0 / (hits[0] + 1)


def _rank_groups(df, score_col, seed=0):
    """Метки в порядке убывания оценки; ничья разрывается случайно.

    В среднее входят только группы с rank_group, не меньше чем из двух товаров
    и хотя бы с одной покупкой. Группа без покупки не даёт нулевой NDCG.
    """
    rng = np.random.default_rng(seed)
    ranked = df[df["rank_group"]]
    for _, g in ranked.groupby(["user_id", "day"], sort=True):
        if len(g) < 2 or int(g["label"].sum()) < 1:
            continue
        # Один и тот же item_id всегда получает один и тот же жребий.
        g = g.sort_values("item_id")
        noise = rng.random(len(g))
        order = np.lexsort((noise, -g[score_col].to_numpy()))
        yield g["label"].to_numpy()[order]


def ndcg_at_k(labels, k):
    gains = labels[:k]
    dcg = (gains / np.log2(np.arange(2, len(gains) + 2))).sum()
    ideal = np.sort(labels)[::-1][:k]
    idcg = (ideal / np.log2(np.arange(2, len(ideal) + 2))).sum()
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(labels, k):
    total = labels.sum()
    return labels[:k].sum() / total if total > 0 else 0.0


def evaluate(df, score_col, is_probability=False):
    """Классификация по всем строкам, ранжирование только по группам с покупкой."""
    res = {"AUC": roc_auc_score(df["label"], df[score_col])}
    if is_probability:
        res["logloss"] = log_loss(df["label"], df[score_col])
    groups = list(_rank_groups(df, score_col))
    for k in (3, 5, 10):
        res[f"NDCG@{k}"] = np.mean([ndcg_at_k(g, k) for g in groups])
    res["Recall@3"] = np.mean([recall_at_k(g, 3) for g in groups])
    res["Recall@5"] = np.mean([recall_at_k(g, 5) for g in groups])
    res["MRR"] = np.mean([reciprocal_rank(g) for g in groups])
    res["groups"] = len(groups)
    return res


if __name__ == "__main__":
    # Попадания на позициях 1 и 4 из 5: ожидаемый NDCG@5 ≈ 0.877
    labels = np.array([1, 0, 0, 1, 0])
    print("NDCG@5 toy:", round(ndcg_at_k(labels, 5), 3))