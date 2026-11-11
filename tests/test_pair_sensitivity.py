"""Small controlled tests for exact pairwise support, confidence and filtering."""
import csv

from src.pair_sensitivity import run


def test_pairwise_rule_metrics_and_outlier_filter(tmp_path):
    path = tmp_path / "baskets.xls"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows([
            ["InvoiceNo", "A", "B", "C"],
            ["1", "True", "True", "False"],
            ["2", "True", "True", "False"],
            ["3", "True", "False", "True"],
            ["4", "False", "True", "True"],
        ])
    result = run(path, min_support=0.5, min_confidence=0.5)
    full = result["populations"]["all"]
    assert full["invoices"] == 4
    assert full["frequent_pairs"] == 1
    assert full["qualifying_pair_rules"] == 2
    a_to_b = next(r for r in full["top_10_by_lift"] if r["antecedent"] == "A")
    assert a_to_b["consequent"] == "B"
    assert a_to_b["support"] == 0.5
    assert abs(a_to_b["confidence"] - 2 / 3) < 1e-12
    assert abs(a_to_b["lift"] - 8 / 9) < 1e-12
    assert result["populations"]["max_100"]["invoices"] == 4
