"""Multiple chronological origins with paired customer-clustered uncertainty.

One concealed training-known product per eligible test invoice; all model
comparisons share the same target and observed basket. Cluster bootstrap
resamples CustomerID, with invoice-level fallback IDs for missing customers.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from src.data import clean_transactions, load_transactions
from src.chronological_holdout import invoice_matrix, rule_statistics
from src.heldout_topk import recommend

def customer_clusters(df, grouped):
    # Missing CustomerID is NOT one giant cluster: assign invoice-specific IDs.
    x = df[["InvoiceNo", "CustomerID"]].drop_duplicates("InvoiceNo")
    ids = dict(zip(x["InvoiceNo"].astype(str), x["CustomerID"]))
    return [("customer:" + str(ids[inv]) if inv in ids and not np.isnan(ids[inv])
             else "invoice:" + inv) for inv in grouped["InvoiceNo"]]

def paired_cluster_interval(difference, clusters, n_bootstrap, seed):
    """Cluster bootstrap of paired invoice-level hit differences."""
    unique, inverse = np.unique(np.asarray(clusters, dtype=str), return_inverse=True)
    sums = np.bincount(inverse, weights=difference, minlength=len(unique))
    sizes = np.bincount(inverse, minlength=len(unique))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(unique), size=(n_bootstrap, len(unique)))
    values = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]

def evaluate(workbook, origins=(0.5, 0.6, 0.7), horizon=0.1,
             k_values=(1, 5, 10), bootstrap=500, seed=2026):
    if bootstrap < 1 or not 0 < horizon < 1:
        raise ValueError("Invalid bootstrap or horizon")
    raw = load_transactions(Path(workbook))
    cleaned, cleaning = clean_transactions(raw)
    grouped, codes, matrix = invoice_matrix(cleaned)
    clusters = customer_clusters(cleaned, grouped)
    n = matrix.shape[0]
    folds = []
    for fold_index, fraction in enumerate(origins):
        cut = int(n * fraction)
        end = min(n, int(n * (fraction + horizon)))
        if not 0 < cut < end <= n:
            raise ValueError("Invalid temporal origin/horizon")
        train, test = matrix[:cut], matrix[cut:end]
        popularity = np.asarray(train.sum(axis=0)).ravel()
        catalogue = set(np.flatnonzero(popularity > 0))
        pair_rules, counts = rule_statistics(train, codes, .01, .2)
        rules = {}
        for a, b, joint in pair_rules:
            rules.setdefault(a, []).append((b, joint / counts[a]))
        rng = np.random.default_rng(seed + fold_index)
        records = {m: {k: [] for k in k_values} for m in ("rules", "popularity", "random")}
        cluster_ids = []
        excluded = 0
        for i in range(test.shape[0]):
            items = set(test.indices[test.indptr[i]:test.indptr[i + 1]])
            candidates = sorted(items & catalogue)
            if len(items) < 2 or len(candidates) < 2:
                excluded += 1
                continue
            target = int(rng.choice(candidates))
            observed = items - {target}
            cluster_ids.append(clusters[cut + i])
            for method in records:
                ranked = recommend(observed, popularity, rules, max(k_values), method, rng)
                for k in k_values:
                    records[method][k].append(int(target in ranked[:k]))
        if not cluster_ids:
            raise ValueError("No eligible test invoices in fold")
        metrics = {}
        for k in k_values:
            a = np.asarray(records["rules"][k], dtype=float)
            b = np.asarray(records["popularity"][k], dtype=float)
            c = np.asarray(records["random"][k], dtype=float)
            metrics[str(k)] = {
                "rules_hit_rate": float(a.mean()),
                "popularity_hit_rate": float(b.mean()),
                "random_hit_rate": float(c.mean()),
                "rules_minus_popularity": float((a-b).mean()),
                "rules_minus_popularity_customer_cluster_95pct_ci": paired_cluster_interval(
                    a-b, cluster_ids, bootstrap, seed + fold_index*100 + k),
                "rules_minus_random": float((a-c).mean()),
                "rules_minus_random_customer_cluster_95pct_ci": paired_cluster_interval(
                    a-c, cluster_ids, bootstrap, seed + fold_index*100 + k + 50),
            }
        folds.append({
            "origin_fraction": fraction, "train_invoices": cut,
            "test_invoices": end-cut, "eligible_test_invoices": len(cluster_ids),
            "excluded_test_invoices": excluded,
            "unique_customer_or_invoice_clusters": len(set(cluster_ids)),
            "train_end": str(grouped.iloc[cut-1]["date"]),
            "test_start": str(grouped.iloc[cut]["date"]),
            "test_end": str(grouped.iloc[end-1]["date"]),
            "train_pair_rules": len(pair_rules),
            "metrics": metrics,
        })
    return {
        "source": "Official UCI Online Retail UK cleaned invoices",
        "protocol": "Expanding-window 50/60/70% origins; each evaluated on following 10% invoices; non-overlapping test windows",
        "seed": seed, "bootstrap_resamples": bootstrap,
        "folds": folds, "cleaning": cleaning,
        "limitations": [
            "CustomerID absent invoices treated as independent invoice clusters; potential unidentified repeated customers.",
            "CustomerID clustering preserves dependence within known customers in each fold.",
            "Cluster bootstrap intervals apply to paired differences, not claims of causal or online uplift.",
            "Three adjacent chronological test windows are not independent external replications.",
            "One concealed train-known item per eligible invoice; cold-start invoices excluded.",
            "No hyperparameter tuning on test windows; support 1%, confidence 20% fixed.",
            "Paired percentile bootstrap intervals are exploratory; no multiplicity-adjusted p-values.",
        ],
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("workbook", type=Path)
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--output", type=Path, default=Path("reports/generated/temporal_clustered.json"))
    args = p.parse_args()
    result = evaluate(args.workbook, bootstrap=args.bootstrap)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"folds": result["folds"]}, indent=2))

if __name__ == "__main__":
    main()
