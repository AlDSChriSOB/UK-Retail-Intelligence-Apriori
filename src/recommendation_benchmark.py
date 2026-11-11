"""Temporal next-item recommendation benchmark for UK invoice baskets.

Protocol: learn item popularity, item co-occurrence and directional Apriori
pair rules on earliest 70% of invoices; evaluate later invoices by hiding
one deterministic item from each basket of at least two distinct products.
The visible remainder is the query context. No test labels inform training.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from src.data import load_transactions, clean_transactions
from src.chronological_holdout import invoice_matrix


def fit_models(train, min_support=0.01, min_confidence=0.20):
    n = train.shape[0]
    freq = np.asarray(train.sum(axis=0)).ravel().astype(np.int64)
    co = (train.astype(np.int64).T @ train.astype(np.int64)).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    # Only training-derived rules are eligible for Apriori ranking.
    rules = co.copy().astype(np.float64)
    for a in range(rules.shape[0]):
        start, end = rules.indptr[a:a + 2]
        if freq[a] == 0:
            rules.data[start:end] = 0
            continue
        joint = rules.data[start:end]
        conf = joint / freq[a]
        eligible = (joint / n >= min_support) & (conf >= min_confidence)
        rules.data[start:end] = np.where(eligible, conf, 0.0)
    rules.eliminate_zeros()
    return freq, co, rules


def rank(context, popularity, cooccurrence, rules, method, k):
    seen = set(context)
    if method == "popularity":
        scores = popularity.astype(float).copy()
    elif method == "cooccurrence":
        scores = np.asarray(cooccurrence[context].sum(axis=0)).ravel().astype(float)
    elif method == "apriori":
        # Sum training-only rule confidences from observed products.
        scores = np.asarray(rules[context].sum(axis=0)).ravel().astype(float)
    else:
        raise ValueError(method)
    scores[list(seen)] = -np.inf
    # Nonpositive association scores are not recommended; popularity
    # fallback makes all methods produce K comparable recommendations.
    if method != "popularity":
        fallback = popularity / max(float(popularity.max()), 1.0)
        positive = scores > 0
        scores = np.where(positive, scores + 1e-6 * fallback, 1e-9 * fallback)
        scores[list(seen)] = -np.inf
    # Stable tie-breaking by product column index.
    order = np.lexsort((np.arange(len(scores)), -scores))
    return order[:k]


def benchmark(matrix, train_fraction=0.7, k_values=(5, 10), seed=42,
              min_support=0.01, min_confidence=0.20):
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    cut = int(matrix.shape[0] * train_fraction)
    if cut < 1 or cut >= matrix.shape[0]:
        raise ValueError("Empty train or test population")
    train, test = matrix[:cut].tocsr(), matrix[cut:].tocsr()
    popularity, cooccurrence, rules = fit_models(train, min_support, min_confidence)
    rng = np.random.default_rng(seed)
    methods = ("popularity", "cooccurrence", "apriori")
    hits = {m: {k: 0 for k in k_values} for m in methods}
    eligible = 0
    test_csr = test.tocsr()
    for i in range(test_csr.shape[0]):
        items = test_csr.indices[test_csr.indptr[i]:test_csr.indptr[i + 1]]
        if len(items) < 2:
            continue
        target_position = int(rng.integers(len(items)))
        target = int(items[target_position])
        context = np.delete(items, target_position)
        eligible += 1
        for method in methods:
            ranked = rank(context, popularity, cooccurrence, rules, method, max(k_values))
            for k in k_values:
                hits[method][k] += int(target in ranked[:k])
    if not eligible:
        raise ValueError("No eligible test baskets")
    # Exactly one hidden item per invoice, so precision@K = hits/(N*K),
    # recall@K = hit_rate@K = hits/N.
    return {
        "protocol": "chronological 70/30 invoice split; one seeded hidden item per eligible test basket",
        "seed": seed, "train_invoices": train.shape[0],
        "test_invoices": test.shape[0], "eligible_test_invoices": eligible,
        "min_support": min_support, "min_confidence": min_confidence,
        "metrics": {
            m: {str(k): {
                "hits": hits[m][k],
                "precision_at_k": hits[m][k] / (eligible * k),
                "recall_at_k": hits[m][k] / eligible,
                "hit_rate_at_k": hits[m][k] / eligible,
            } for k in k_values} for m in methods
        },
        "limitations": [
            "Single hidden product per test basket: recall@K equals hit rate@K.",
            "Within-invoice masking is not next-session or real-time recommendation.",
            "Customers can occur in both periods; no customer-disjoint holdout.",
            "Apriori here uses two-item directional rules, not longer itemsets.",
            "Product codes are sorted, with deterministic tie breaking.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook")
    parser.add_argument("--output", default="reports/generated/recommendation_benchmark.json")
    args = parser.parse_args()
    data, cleaning = clean_transactions(load_transactions(Path(args.workbook)))
    grouped, codes, matrix = invoice_matrix(data)
    result = benchmark(matrix)
    result["cleaning"] = cleaning
    result["product_count"] = len(codes)
    result["last_train_date"] = str(grouped.iloc[result["train_invoices"] - 1]["date"])
    result["first_test_date"] = str(grouped.iloc[result["train_invoices"]]["date"])
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
