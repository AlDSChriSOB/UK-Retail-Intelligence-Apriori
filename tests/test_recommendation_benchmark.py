"""Deterministic recommendation benchmark sanity checks."""
import numpy as np
from scipy.sparse import csr_matrix
from src.recommendation_benchmark import benchmark, fit_models, rank


def test_ranking_excludes_observed_products():
    train = csr_matrix(np.array([[1, 1, 0], [1, 0, 1], [1, 1, 1]], dtype=np.int32))
    freq, co, rules = fit_models(train, min_support=0.2, min_confidence=0.2)
    for method in ("popularity", "cooccurrence", "apriori"):
        ranked = rank(np.array([0]), freq, co, rules, method, 2)
        assert 0 not in ranked
        assert len(ranked) == 2


def test_metrics_with_one_hidden_item():
    # First three rows train; final two are evaluated.
    matrix = csr_matrix(np.array([
        [1, 1, 0], [1, 1, 0], [1, 0, 1],
        [1, 1, 0], [1, 0, 1],
    ], dtype=np.int32))
    result = benchmark(matrix, train_fraction=0.6, k_values=(1, 2), seed=42,
                       min_support=0.2, min_confidence=0.2)
    assert result["eligible_test_invoices"] == 2
    for method in ("popularity", "cooccurrence", "apriori"):
        for k in (1, 2):
            metrics = result["metrics"][method][str(k)]
            assert metrics["recall_at_k"] == metrics["hit_rate_at_k"]
            assert metrics["precision_at_k"] == metrics["hits"] / (2 * k)
