import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src import real_data_apriori_experiment as experiment


def test_bounded_experiment_without_external_dataset(monkeypatch, tmp_path):
    # Fixture only: this test checks plumbing, NOT a real-UCI result.
    rows = [
        {"InvoiceNo": "1", "StockCode": "A", "InvoiceDate": "2020-01-01"},
        {"InvoiceNo": "1", "StockCode": "B", "InvoiceDate": "2020-01-01"},
        {"InvoiceNo": "2", "StockCode": "A", "InvoiceDate": "2020-01-02"},
        {"InvoiceNo": "2", "StockCode": "B", "InvoiceDate": "2020-01-02"},
        {"InvoiceNo": "3", "StockCode": "A", "InvoiceDate": "2020-01-03"},
        {"InvoiceNo": "3", "StockCode": "C", "InvoiceDate": "2020-01-03"},
        {"InvoiceNo": "4", "StockCode": "A", "InvoiceDate": "2020-01-04"},
        {"InvoiceNo": "4", "StockCode": "B", "InvoiceDate": "2020-01-04"},
    ]
    df = pd.DataFrame(rows)
    monkeypatch.setattr(experiment, "load_transactions", lambda _: df)
    monkeypatch.setattr(experiment, "clean_transactions", lambda x: (x, {"final_rows": len(x)}))
    report = experiment.run(tmp_path / "fixture.xlsx", max_invoices=2, min_support=0.5, max_len=3)
    assert report["sample_invoices"] == 2
    assert report["itemsets_and_support_match"] is True
    assert report["independent_itemsets"] == report["mlxtend_itemsets"]
    assert report["independent_performance"]["seconds"] >= 0
    assert report["mlxtend_performance"]["python_tracemalloc_peak_mib"] >= 0
