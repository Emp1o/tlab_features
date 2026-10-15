import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


def _rank_groups(df, score_col, seed=0):
    """Yield label arrays ordered by score; ties are broken randomly."""
    rng = np.random.default_rng(seed)
    for _, g in df[df["rank_group"]].groupby(["user_id", "day"]):
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
    """Classification metrics on all rows, ranking metrics on rank groups."""
    res = {"AUC": roc_auc_score(df["label"], df[score_col])}
    if is_probability:
        res["logloss"] = log_loss(df["label"], df[score_col])
    groups = list(_rank_groups(df, score_col))
    for k in (5, 10):
        res[f"NDCG@{k}"] = np.mean([ndcg_at_k(g, k) for g in groups])
    res["Recall@5"] = np.mean([recall_at_k(g, 5) for g in groups])
    res["groups"] = len(groups)
    return res


if __name__ == "__main__":
    # Hits at positions 1 and 4 out of 5: expected NDCG@5 ≈ 0.877
    labels = np.array([1, 0, 0, 1, 0])
    print("NDCG@5 toy:", round(ndcg_at_k(labels, 5), 3))