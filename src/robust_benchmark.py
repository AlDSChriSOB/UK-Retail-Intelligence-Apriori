"""Leakage-controlled sensitivity benchmark with co-occurrence and cosine baselines.

The same held-out target and context are shared by all methods per invoice.
No test data are used for model fitting or threshold selection.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from src.data import load_transactions, clean_transactions
from src.chronological_holdout import invoice_matrix
from src.recommendation_benchmark import fit_models
from src.temporal_clustered import customer_clusters, paired_cluster_interval

METHODS = ("apriori", "cooccurrence", "cosine", "popularity")

def rank_items(context, freq, co, rules, method, k):
    if method == "popularity":
        scores = freq.astype(float).copy()
    elif method == "apriori":
        scores = np.asarray(rules[list(context)].sum(axis=0)).ravel().astype(float)
    elif method == "cooccurrence":
        scores = np.asarray(co[list(context)].sum(axis=0)).ravel().astype(float)
    elif method == "cosine":
        # Binary invoice-item cosine: co(a,b)/sqrt(freq(a)*freq(b)).
        cols = np.asarray(co[list(context)].toarray(), dtype=float)
        denominators = np.sqrt(np.maximum(freq[list(context), None], 1) * np.maximum(freq[None, :], 1))
        scores = (cols / denominators).sum(axis=0)
    else:
        raise ValueError(method)
    # No unseen training products; all methods use identical popularity fallback.
    valid = freq > 0
    positive = scores > 0
    fallback = freq / max(float(freq.max()), 1.)
    if method != "popularity":
        scores = np.where(positive, scores + 1e-6 * fallback, 1e-9 * fallback)
    scores = np.where(valid, scores, -np.inf)
    scores[list(context)] = -np.inf
    order = np.lexsort((np.arange(len(freq)), -scores))
    return [int(i) for i in order if np.isfinite(scores[i])][:k]

def evaluate(workbook, origins=(.5,.6,.7), horizon=.1, supports=(.005,.01,.02),
             confidences=(.1,.2,.3), k_values=(5,10), bootstrap=300, seed=2026):
    cleaned, cleaning = clean_transactions(load_transactions(Path(workbook)))
    grouped, codes, matrix = invoice_matrix(cleaned)
    clusters = customer_clusters(cleaned, grouped)
    n = matrix.shape[0]
    folds = []
    for fold_index, origin in enumerate(origins):
        cut, end = int(n*origin), int(n*(origin+horizon))
        if not 0 < cut < end <= n:
            raise ValueError("Invalid chronological split")
        train, test = matrix[:cut], matrix[cut:end]
        freq = np.asarray(train.sum(axis=0)).ravel()
        rng = np.random.default_rng(seed+fold_index)
        examples = []
        for i in range(test.shape[0]):
            items = set(test.indices[test.indptr[i]:test.indptr[i+1]])
            known = sorted(j for j in items if freq[j] > 0)
            if len(known) < 2:
                continue
            target = int(rng.choice(known))
            examples.append((items-{target}, target, clusters[cut+i]))
        if not examples:
            raise ValueError("No eligible invoices")
        fold = {"origin":origin, "train_invoices":cut, "test_invoices":end-cut,
                "eligible":len(examples), "clusters":len(set(e[2] for e in examples)),
                "train_end":str(grouped.iloc[cut-1]["date"]),
                "test_start":str(grouped.iloc[cut]["date"]), "configurations":[]}
        for support in supports:
            for confidence in confidences:
                frequency, co, rules = fit_models(train, support, confidence)
                outcomes = {m:{k:[] for k in k_values} for m in METHODS}
                for context,target,_ in examples:
                    for method in METHODS:
                        ranked = rank_items(context, frequency, co, rules, method, max(k_values))
                        for k in k_values:
                            outcomes[method][k].append(int(target in ranked[:k]))
                metrics = {}
                cluster_ids = [e[2] for e in examples]
                for k in k_values:
                    metrics[str(k)] = {m:float(np.mean(outcomes[m][k])) for m in METHODS}
                    for competitor in ("cooccurrence","cosine","popularity"):
                        diff = np.asarray(outcomes["apriori"][k],float)-np.asarray(outcomes[competitor][k],float)
                        metrics[str(k)]["apriori_minus_"+competitor] = {
                            "difference":float(diff.mean()),
                            "customer_cluster_95pct_ci":paired_cluster_interval(
                                diff, cluster_ids, bootstrap, seed+fold_index*10000+
                                int(support*100000)*10+int(confidence*100)+k+
                                METHODS.index(competitor))
                        }
                fold["configurations"].append({"min_support":support,"min_confidence":confidence,
                    "rule_nonzero_weights":int(rules.nnz),"metrics":metrics})
        folds.append(fold)
    return {"source":"Official UCI Online Retail; UK positive non-cancelled invoices",
        "protocol":"Expanding 50/60/70% origins, following non-overlapping 10% windows; identical hidden target per fold across all configurations",
        "methods":list(METHODS),"supports":list(supports),"confidences":list(confidences),
        "k_values":list(k_values),"bootstrap":bootstrap,"seed":seed,"folds":folds,
        "cleaning":cleaning,"limitations":[
            "All 27 fold/configuration combinations are exploratory sensitivity comparisons, not hyperparameter selection.",
            "Bootstrap intervals are unadjusted for multiple comparisons.",
            "Known CustomerID clustered within fold; missing IDs treated as separate invoice clusters.",
            "Three neighboring time windows do not provide independent external validation.",
            "Single hidden train-known product measures basket completion, not future purchase.",
            "No revenue uplift or online impact can be inferred."
        ]}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("workbook",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/robust_benchmark.json"))
    p.add_argument("--bootstrap",type=int,default=300)
    a=p.parse_args()
    result=evaluate(a.workbook,bootstrap=a.bootstrap)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps({"folds":[{"origin":f["origin"],"eligible":f["eligible"],"configurations":len(f["configurations"])} for f in result["folds"]]},indent=2))
if __name__=="__main__":
    main()
