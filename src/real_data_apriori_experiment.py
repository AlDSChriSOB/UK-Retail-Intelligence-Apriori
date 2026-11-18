"""Bounded real-UCI Apriori experiment with independent mlxtend comparison.

No metrics are assumed. Outputs are created only after successful execution.
"""
import argparse
import json
import platform
import time
import tracemalloc
from pathlib import Path

import pandas as pd
from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from src.data import load_transactions, clean_transactions
from src.chronological_holdout import invoice_matrix
from src.full_apriori import apriori as independent_apriori


def measure(fn):
    tracemalloc.start()
    start = time.perf_counter()
    try:
        result = fn()
        elapsed = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return result, {"seconds": elapsed, "python_tracemalloc_peak_mib": peak / (1024 ** 2)}


def run(workbook, max_invoices=1000, min_support=0.05, max_len=3):
    if max_invoices < 2:
        raise ValueError("max_invoices must be >= 2")
    if not 0 < min_support <= 1:
        raise ValueError("min_support must be in (0,1]")
    raw = load_transactions(Path(workbook))
    cleaned, cleaning = clean_transactions(raw)
    grouped, codes, matrix = invoice_matrix(cleaned)
    cut = int(matrix.shape[0] * 0.7)
    if cut < 2:
        raise ValueError("Insufficient training invoices")
    # Deterministic, earliest chronological training invoices; never use test invoices.
    sample = matrix[:min(cut, max_invoices)].tocsr()
    # Restrict catalogue to products observed in the selected training sample.
    active = sample.getnnz(axis=0) > 0
    sample = sample[:, active].astype(bool)
    selected_codes = [str(code) for code, use in zip(codes, active) if use]
    baskets = [
        {selected_codes[col] for col in sample.indices[sample.indptr[i]:sample.indptr[i + 1]]}
        for i in range(sample.shape[0])
    ]
    independent, independent_perf = measure(
        lambda: independent_apriori(baskets, min_support=min_support, max_len=max_len)
    )
    frame = pd.DataFrame.sparse.from_spmatrix(sample, columns=selected_codes).astype(bool)
    reference, reference_perf = measure(
        lambda: mlxtend_apriori(frame, min_support=min_support, use_colnames=True, max_len=max_len, low_memory=True)
    )
    expected = {frozenset(row.itemsets): float(row.support) for row in reference.itertuples()}
    matching = independent.keys() == expected.keys() and all(
        abs(value - expected[key]) <= 1e-10 for key, value in independent.items()
    )
    result = {
        "source": "UCI Online Retail; cleaned UK transactions",
        "protocol": "first N chronological training invoices from 70 percent train split; test excluded",
        "sample_invoices": sample.shape[0],
        "sample_products": sample.shape[1],
        "train_invoices_available": cut,
        "min_support": min_support,
        "max_len": max_len,
        "independent_itemsets": len(independent),
        "mlxtend_itemsets": len(expected),
        "itemsets_and_support_match": matching,
        "independent_performance": independent_perf,
        "mlxtend_performance": reference_perf,
        "memory_measurement_note": "tracemalloc measures Python-tracked allocations, not total process RSS",
        "python_version": platform.python_version(),
        "cleaning": cleaning,
    }
    if not matching:
        raise AssertionError("Apriori and mlxtend frequent itemsets/support differ")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--max-invoices", type=int, default=1000)
    parser.add_argument("--min-support", type=float, default=0.05)
    parser.add_argument("--max-len", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("reports/generated/full_apriori_real_data.json"))
    args = parser.parse_args()
    result = run(args.workbook, args.max_invoices, args.min_support, args.max_len)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
