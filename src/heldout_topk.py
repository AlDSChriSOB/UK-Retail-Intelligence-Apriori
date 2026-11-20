"""Chronological, leakage-controlled basket-completion recommendation evaluation.

Train-only directional pair rules versus train-only popularity and deterministic random.
For each eligible test invoice, hide one product; recommend from remaining products.
Metrics are per eligible invoice. No customer-personalization or online CTR claims.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from src.data import load_transactions, clean_transactions
from src.chronological_holdout import invoice_matrix, rule_statistics

def recommend(observed, popularity, rules, k, method, rng=None):
    seen = set(observed)
    if method == "rules":
        scores = {}
        for a in seen:
            for b, score in rules.get(a, ()):
                if b not in seen:
                    scores[b] = max(scores.get(b, 0.0), score)
        ranked = sorted(scores, key=lambda b: (-scores[b], -popularity[b], b))
        # Popularity fallback ensures K recommendations when catalogue permits.
        ranked += [b for b in np.argsort(-popularity, kind="stable") if b not in seen and b not in scores and popularity[b] > 0]
    elif method == "popularity":
        ranked = [int(b) for b in np.argsort(-popularity, kind="stable") if b not in seen and popularity[b] > 0]
    else:
        ranked = [int(b) for b in np.flatnonzero(popularity > 0) if b not in seen]
        rng.shuffle(ranked)
    return ranked[:k]

def evaluate(workbook, k_values=(1, 5, 10), seed=2026, bootstrap=500,
             min_support=0.01, min_confidence=0.2):
    if bootstrap < 1 or not k_values or any(k < 1 for k in k_values):
        raise ValueError("Bootstrap and K values must be positive")
    raw = load_transactions(Path(workbook))
    cleaned, cleaning = clean_transactions(raw)
    grouped, codes, matrix = invoice_matrix(cleaned)
    cut = int(matrix.shape[0] * 0.7)
    train, test = matrix[:cut], matrix[cut:]
    popularity = np.asarray(train.sum(axis=0)).ravel()
    pairs, counts = rule_statistics(train, codes, min_support, min_confidence)
    rules = {}
    for a, b, both in pairs:
        rules.setdefault(a, []).append((b, both / counts[a]))
    rng = np.random.default_rng(seed)
    methods = ("rules", "popularity", "random")
    max_k = max(k_values)
    hits = {method: {k: [] for k in k_values} for method in methods}
    recommended = {method: {k: set() for k in k_values} for method in methods}
    eligible = 0
    excluded_single = excluded_unseen = 0
    catalogue = set(np.flatnonzero(popularity > 0))
    for i in range(test.shape[0]):
        items = test.indices[test.indptr[i]:test.indptr[i + 1]].tolist()
        if len(items) < 2:
            excluded_single += 1
            continue
        known = sorted(set(items) & catalogue)
        if len(known) < 2:
            excluded_unseen += 1
            continue
        target = int(rng.choice(known))
        observed = set(items) - {target}
        eligible += 1
        for method in methods:
            ranked = recommend(observed, popularity, rules, max_k, method, rng)
            for k in k_values:
                top = ranked[:k]
                hits[method][k].append(int(target in top))
                recommended[method][k].update(top)
    if not eligible:
        raise ValueError("No eligible held-out invoices")
    ci_rng = np.random.default_rng(seed + 1)
    resample = ci_rng.integers(0, eligible, size=(bootstrap, eligible))
    results = {}
    for method in methods:
        results[method] = {}
        for k in k_values:
            values = np.asarray(hits[method][k], dtype=float)
            rates = values[resample].mean(axis=1)
            results[method][str(k)] = {
                "hit_rate_at_k": float(values.mean()),
                "recall_at_k": float(values.mean()),
                "precision_at_k": float(values.mean() / k),
                "hit_rate_95pct_percentile_ci": [float(x) for x in np.quantile(rates, [0.025, 0.975])],
                "catalogue_coverage_at_k": len(recommended[method][k]) / len(catalogue),
            }
    return {
        "protocol": "70/30 chronological invoice split; one seeded randomly held-out TRAIN-KNOWN item per eligible test basket; observed remaining items",
        "source": "Official UCI Online Retail, cleaned UK positive non-cancelled sales",
        "seed": seed, "bootstrap_resamples": bootstrap, "k_values": list(k_values),
        "train_invoices": cut, "test_invoices": test.shape[0],
        "train_end": str(grouped.iloc[cut - 1]["date"]),
        "test_start": str(grouped.iloc[cut]["date"]),
        "eligible_test_invoices": eligible,
        "excluded_single_item_invoices": excluded_single,
        "excluded_insufficient_train_known_items": excluded_unseen,
        "train_catalogue_size": len(catalogue),
        "train_pair_rules": len(pairs),
        "min_support": min_support, "min_confidence": min_confidence,
        "results": results, "cleaning": cleaning,
        "limitations": [
            "One withheld product per eligible invoice: recall@K equals hit rate@K; precision@K equals hit rate@K/K.",
            "Eligible test invoices have at least two products known in training; cold-start invoices excluded.",
            "Random masking within a completed invoice is basket completion, not next-purchase forecasting.",
            "Random baseline sampled per invoice from train-known products; fixed seed.",
            "Bootstrap intervals resample invoices independently; repeated customers may violate independence.",
            "Confidence scores are max across observed antecedents, with popularity fallback.",
            "No parameter tuning on test; one fixed train/test period; not a claim of real-world conversion uplift.",
        ],
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("workbook", type=Path)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--output", type=Path, default=Path("reports/generated/heldout_topk.json"))
    args = p.parse_args()
    report = evaluate(args.workbook, bootstrap=args.bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("eligible_test_invoices", "train_catalogue_size", "train_pair_rules", "results")}, indent=2))

if __name__ == "__main__":
    main()
