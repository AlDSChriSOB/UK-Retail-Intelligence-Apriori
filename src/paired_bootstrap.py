"""Paired bootstrap uncertainty for masked-basket recommender comparisons.

Consumes a CSV with one row per test invoice and 0/1 hit columns,
e.g. apriori_at_10, cooccurrence_at_10, popularity_at_10.
Bootstrap resamples *invoices*, preserving paired method outcomes.
"""
from __future__ import annotations
import argparse
import csv
import json
import numpy as np


def paired_bootstrap(a, b, replicates=10000, seed=20261111):
    a = np.asarray(a, dtype=np.int8)
    b = np.asarray(b, dtype=np.int8)
    if a.ndim != 1 or a.shape != b.shape or not len(a):
        raise ValueError("Expected nonempty, equally sized 1-D paired outcomes")
    if not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Outcomes must be binary")
    if replicates < 100:
        raise ValueError("Use at least 100 bootstrap replicates")
    diff = a.astype(np.float64) - b.astype(np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(replicates)
    for i in range(replicates):
        boot[i] = diff[rng.integers(0, len(diff), len(diff))].mean()
    low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "n": int(len(diff)),
        "hit_rate_a": float(a.mean()),
        "hit_rate_b": float(b.mean()),
        "difference": float(diff.mean()),
        "ci95_percentile": [float(low), float(high)],
        "a_only": int(((a == 1) & (b == 0)).sum()),
        "b_only": int(((a == 0) & (b == 1)).sum()),
        "both_hit": int(((a == 1) & (b == 1)).sum()),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Per-invoice 0/1 hit outcomes CSV")
    parser.add_argument("--a", default="apriori_at_10")
    parser.add_argument("--b", default="cooccurrence_at_10")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261111)
    args = parser.parse_args()
    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = paired_bootstrap([int(row[args.a]) for row in rows],
                              [int(row[args.b]) for row in rows],
                              args.replicates, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
