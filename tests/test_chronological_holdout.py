"""Unit tests for leakage-aware invoice chronology and rule statistics."""
import pandas as pd
from src.chronological_holdout import invoice_matrix, rule_statistics


def test_invoice_order_and_pair_rule_metrics():
    df = pd.DataFrame({
        "InvoiceNo": ["later", "early", "early", "mid", "mid"],
        "StockCode": ["B", "A", "B", "A", "B"],
        "InvoiceDate": pd.to_datetime([
            "2020-01-03", "2020-01-01", "2020-01-01",
            "2020-01-02", "2020-01-02",
        ]),
    })
    grouped, codes, matrix = invoice_matrix(df)
    assert grouped["InvoiceNo"].tolist() == ["early", "mid", "later"]
    assert matrix.shape == (3, 2)
    rules, counts = rule_statistics(matrix, codes, min_support=0.5, min_confidence=0.5)
    assert len(rules) == 2
    assert set((codes[a], codes[b], both) for a, b, both in rules) == {
        ("A", "B", 2), ("B", "A", 2),
    }
    assert counts.tolist() == [2, 3]
