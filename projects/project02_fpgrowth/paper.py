"""Generate evidence-backed manuscript and publication figures; no placeholder results."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def build(resource_path, recommendation_path, out_dir):
    resource=json.loads(Path(resource_path).read_text())
    rec=json.loads(Path(recommendation_path).read_text())
    folds=rec["folds"]
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    x=np.arange(len(folds))
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    for method in ("apriori","cosine","popularity"):
        ax.plot(x,[100*f["metrics"]["10"][method] for f in folds],marker="o",label=method.title())
    ax.plot(x,[100*f["metrics"]["10"]["fpgrowth"] for f in folds],marker="s",
            linestyle="--",linewidth=2.0,color="black",zorder=10,
            label="FP-Growth (identical to Apriori)")
    ax.set_xticks(x,[f"{int(f['origin']*100)}% train" for f in folds])
    ax.set_ylabel("Hit Rate@10 (%)")
    ax.set_xlabel("Chronological training origin")
    ax.set_title("Matched-miner accuracy and competing baselines")
    ax.legend(ncol=2,frameon=False,loc="upper center",bbox_to_anchor=(.5,-.18));ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(out/f"figure_1_accuracy.{ext}",dpi=240)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.4))
    groups=np.arange(len(resource["results"]))
    width=.37
    ax.bar(groups-width/2,[r["apriori_seconds"] for r in resource["results"]],width,label="Apriori")
    ax.bar(groups+width/2,[r["fpgrowth_seconds"] for r in resource["results"]],width,label="FP-Growth")
    ax.set_xticks(groups,[f"{int(r['train_origin']*100)}% / {r['min_support']:.2f}" for r in resource["results"]],rotation=30,ha="right")
    ax.set_ylabel("Single-run elapsed time (seconds)")
    ax.set_xlabel("Training origin / minimum support")
    ax.set_title("Exploratory mining runtime")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("pdf","png"):
        fig.savefig(out/f"figure_2_runtime.{ext}",dpi=240)
    plt.close(fig)
    rows=[]
    for fold in folds:
        m=fold["metrics"]["10"]
        d=m["fpgrowth_minus_cosine"]
        rows.append(f"| {int(fold['origin']*100)}% | {fold['eligible']:,} | {m['fpgrowth']:.2%} | {m['apriori']:.2%} | {m['cosine']:.2%} | {m['popularity']:.2%} | {100*d['difference']:+.2f} | [{100*d['customer_cluster_95pct_ci'][0]:.2f}, {100*d['customer_cluster_95pct_ci'][1]:.2f}] |")
    costs=[]
    for r in resource["results"]:
        costs.append(f"| {int(r['train_origin']*100)}% | {r['min_support']:.3f} | {r['apriori_itemsets']:,} | {r['apriori_seconds']:.3f} | {r['fpgrowth_seconds']:.3f} | {r['fpgrowth_speedup_ratio']:.2f} |")
    all_equal=all(f["exact_itemset_equivalence"] and f["exact_prediction_equivalence"] for f in folds)
    resource_equal=all(r["equal_itemsets_and_support"] for r in resource["results"])
    if not all_equal or not resource_equal:
        raise AssertionError("Cannot publish manuscript: equivalence verification failed")
    text=f"""# FP-Growth versus Apriori: computational cost and matched recommendation accuracy on chronological UK retail invoices

**Status:** Evidence-generated research manuscript; journal selection and author declarations outstanding.

## Abstract
**Background:** FP-Growth and Apriori are alternative algorithms for frequent-itemset mining. **Objective:** Compare their mining costs and determine whether equivalent itemsets yield equivalent basket-completion recommendations. **Methods:** Official UCI Online Retail transactions were cleaned to positive, non-cancelled UK invoices. Two-item frequent sets were mined from chronological training prefixes at 50%, 60% and 70%; supports 0.01 and 0.02 were assessed for computational cost. At support 0.01 and confidence 0.20, matched directional pair-confidence recommendations were evaluated on the following 10% of invoices, hiding one train-known product per eligible basket. Cosine similarity and popularity served as comparators. Customer-clustered paired percentile bootstrap intervals used {rec['bootstrap']} resamples. **Results:** All {len(resource['results'])} cost conditions had identical frequent itemsets and support values across miners; all {len(folds)} recommendation folds had identical FP-Growth and Apriori predictions. Cosine and popularity performance, fold sizes and paired differences are given in Table 1. **Conclusions:** FP-Growth changes the itemset-mining procedure but, under identical itemsets and scoring, cannot itself improve recommendation accuracy. Runtime comparisons are exploratory and hardware-dependent; no online recommendation uplift is established.

**Keywords:** FP-Growth; Apriori; frequent itemsets; market basket; recommender systems; chronological evaluation; bootstrap

## 1. Introduction
The Apriori algorithm uses candidate generation to discover frequent itemsets [2], while FP-Growth uses a compressed frequent-pattern tree and pattern growth [3]. Faster mining should not be confused with more accurate ranking when the resulting rule sets are identical. This paper separates algorithmic equivalence and mining cost from recommendation performance.

## 2. Materials and methods
### 2.1 Dataset and provenance
We use the genuine UCI Online Retail dataset [1]. Rows lacking invoice, product code or date were excluded, as were cancellation invoices, non-positive quantity or price, and non-UK transactions. Duplicate invoice–product occurrences were binarized. The repository records exact cleaning code and downloads the workbook from the official archive during CI.

### 2.2 Frequent-itemset equivalence and computational cost
Training-only binary invoice–product matrices were created from the first 50%, 60% and 70% of chronologically ordered invoices. FP-Growth and Apriori from mlxtend mined itemsets of maximum length two at minimum supports 0.01 and 0.02. Itemset identities were required to match exactly, and supports within 1e-10. Single-process wall-clock timings and tracemalloc Python-allocation peaks were recorded. These are not controlled multi-run performance measurements.

### 2.3 Basket-completion task
For each training origin, the next 10% of invoices formed the held-out test window. Eligible invoices had at least two distinct training-known products; one was hidden with a deterministic seeded generator. Remaining products were the shared query context. This is within-invoice reconstruction, not future-order prediction.

### 2.4 Scoring and baselines
Frequent two-item sets were converted into directional rules using confidence P(B|A), retaining confidence >=0.20. Candidate scores summed confidence over observed antecedents. Both miners used the same rule extraction, popularity fallback, candidate exclusions and deterministic tie-breaking. Cosine-normalized binary co-occurrence and training-item popularity were evaluated against the same concealed targets.

### 2.5 Metrics and uncertainty
Hit Rate@5 and Hit Rate@10 were evaluated. Exactly one product was hidden per invoice, so Recall@K equals Hit Rate@K. Paired FP-Growth-minus-comparator differences used {rec['bootstrap']} customer-cluster percentile bootstrap replicates within each fold; missing customer identifiers were treated as invoice-specific clusters. Intervals are exploratory and unadjusted for multiple comparisons.

## 3. Results
### 3.1 Recommendation accuracy
**Table 1.** Matched recommendation Hit Rate@10, with paired FP-Growth minus cosine difference and 95% cluster-bootstrap interval in percentage points.

| Training origin | Eligible invoices | FP-Growth | Apriori | Cosine | Popularity | FP minus cosine (pp) | 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

FP-Growth and Apriori predictions were exactly identical across all eligible evaluation invoices and both evaluated K values. Their paired difference is therefore zero by construction and verification, not an independent empirical superiority finding.

### 3.2 Frequent-itemset equivalence and mining runtime
**Table 2.** Itemset counts and one-run elapsed time. Speedup is Apriori seconds divided by FP-Growth seconds; values are hardware- and order-dependent.

| Training origin | Support | Matching itemsets | Apriori (s) | FP-Growth (s) | Speedup ratio |
|---|---:|---:|---:|---:|---:|
{chr(10).join(costs)}

### 3.3 Figures
**Figure 1.** Held-out Hit Rate@10 by method across three chronological test windows. The FP-Growth and Apriori curves overlap by verified prediction equivalence.

**Figure 2.** Single-run elapsed time for FP-Growth and Apriori across training prefixes and support thresholds. This chart does not establish statistical runtime superiority.

## 4. Discussion
The verified equality of frequent itemsets implies equality of pair-confidence rules under the same filtering and scoring. FP-Growth and Apriori are alternative discovery algorithms, not inherently different recommendation models. Cosine and popularity comparisons quantify the ranking quality of these mined rules under a constrained retrospective evaluation. Computational cost should be evaluated separately with repeated randomized-order timings before generalizing about speed.

## 5. Limitations
(1) Historical UK retail data may not represent modern or non-UK demand. (2) Concealed-item recovery is not next-order prediction or online uplift. (3) Only frequent itemsets of size at most two were mined. (4) No repeated timing trials or process-RSS measurements were performed. (5) Known customers may recur across folds; missing customer identifiers weaken dependence adjustment. (6) Bootstrap intervals are exploratory, not multiplicity-adjusted. (7) Training-known-item eligibility excludes cold-start cases. (8) No revenue or causal conversion impact was measured.

## 6. Conclusions
FP-Growth and Apriori generated equivalent frequent itemsets and identical pair-rule recommendations in the tested settings. Their accuracy must be interpreted as the same model expressed through two mining implementations. This study supplies a reproducible separation of computational and predictive performance, without asserting that mining speed causes better recommendations.

## Data and code availability
UCI Online Retail dataset: https://archive.ics.uci.edu/dataset/352/online+retail . Code: https://github.com/AlDSChriSOB/UK-Retail-Intelligence-Apriori under `projects/project02_fpgrowth`. CI artifacts include the two evidence JSON files and the publication figures. Dataset is not redistributed.

## Declarations requiring author confirmation
Funding, conflicts of interest, author contributions, affiliation, corresponding author and applicable ethics/AI-assistance disclosures must be confirmed before submission. No approval or journal acceptance is implied.

## References
1. Chen, D. (2015). *Online Retail* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
2. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules in large databases. *Proceedings of the 20th International Conference on Very Large Data Bases*, 487–499. https://www.vldb.org/conf/1994/P487.PDF
3. Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. *Proceedings of the 2000 ACM SIGMOD International Conference on Management of Data*, 1–12. https://doi.org/10.1145/342009.335372
4. Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The Annals of Statistics, 7*(1), 1–26. https://doi.org/10.1214/aos/1176344552
"""
    (out/"PROJECT02_MANUSCRIPT.md").write_text(text,encoding="utf-8")
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("resource_json",type=Path)
    p.add_argument("recommendation_json",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project02_paper"))
    a=p.parse_args()
    print(build(a.resource_json,a.recommendation_json,a.output))
if __name__=="__main__":
    main()
