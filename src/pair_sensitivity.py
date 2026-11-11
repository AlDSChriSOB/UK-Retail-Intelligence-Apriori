"""Reproduce exact two-item association-rule sensitivity on a binary basket CSV.

Usage: python -m src.pair_sensitivity path/to/basket.xls --output reports/generated/pair_sensitivity.json
The input may have a .xls extension but MUST be CSV-formatted text.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix


def run(path, min_support=0.01, min_confidence=0.20):
    rows, cols, sizes = [], [], []
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        codes = header[1:]
        if not codes or len(set(codes)) != len(codes):
            raise ValueError("Missing or duplicate product columns")
        for i, row in enumerate(reader):
            if len(row) != len(header):
                raise ValueError(f"Malformed basket row {i+2}")
            active = []
            for j, value in enumerate(row[1:]):
                v = value.strip().lower()
                if v in ("true", "1"):
                    active.append(j)
                elif v not in ("false", "0"):
                    raise ValueError(f"Nonbinary value at row {i+2}, product {codes[j]}")
            if not active:
                raise ValueError(f"Empty basket at row {i+2}")
            rows.extend([i] * len(active))
            cols.extend(active)
            sizes.append(len(active))
    n, m = len(sizes), len(codes)
    if not n:
        raise ValueError("No invoices")
    matrix = csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(n, m))
    sizes = np.asarray(sizes)
    output = {"min_support": min_support, "min_confidence": min_confidence, "populations": {}}
    full_keys = set()
    for name, mask in (("all", np.ones(n, bool)), ("max_250", sizes <= 250), ("max_100", sizes <= 100)):
        subset = matrix[mask]
        total = subset.shape[0]
        counts = np.asarray(subset.sum(axis=0)).ravel()
        selected = np.flatnonzero(counts >= np.ceil(min_support * total))
        cooccurrence = (subset[:, selected].T @ subset[:, selected]).toarray()
        rules = []
        pair_count = 0
        for a in range(len(selected)):
            for b in range(a + 1, len(selected)):
                both = int(cooccurrence[a, b])
                if both / total < min_support:
                    continue
                pair_count += 1
                for u, v in ((a, b), (b, a)):
                    confidence = both / int(counts[selected[u]])
                    if confidence >= min_confidence:
                        rules.append({
                            "antecedent": codes[selected[u]],
                            "consequent": codes[selected[v]],
                            "support": both / total,
                            "confidence": confidence,
                            "lift": confidence / (counts[selected[v]] / total),
                            "joint_invoices": both,
                        })
        rules.sort(key=lambda x: (-x["lift"], -x["support"], x["antecedent"], x["consequent"]))
        keys = {(r["antecedent"], r["consequent"]) for r in rules}
        result = {
            "invoices": total,
            "excluded": n - total,
            "frequent_singletons": len(selected),
            "frequent_pairs": pair_count,
            "qualifying_pair_rules": len(rules),
            "top_10_by_lift": rules[:10],
        }
        if name == "all":
            full_keys = keys
        else:
            result["rule_overlap_with_full"] = len(keys & full_keys)
        output["populations"][name] = result
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basket")
    parser.add_argument("--output", default="reports/generated/pair_sensitivity.json")
    args = parser.parse_args()
    result = run(args.basket)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: {a: b for a, b in v.items() if a != "top_10_by_lift"} for k, v in result["populations"].items()}, indent=2))


if __name__ == "__main__":
    main()
