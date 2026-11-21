"""Project 03: generate evidence-only figures and research manuscript."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def build(mining_path,recommendation_path,output):
    mining=json.loads(Path(mining_path).read_text())
    rec=json.loads(Path(recommendation_path).read_text())
    rows=mining["results"];folds=rec["folds"]
    if not all(r["exact_three_way_equivalence"] for r in rows):
        raise AssertionError("Mining equivalence failed")
    if not all(f["three_way_itemset_and_prediction_equivalence"] for f in folds):
        raise AssertionError("Recommendation equivalence failed")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(7.8,4.7))
    x=np.arange(len(folds))
    for method in ("cosine","popularity","apriori"):
        ax.plot(x,[100*f["metrics"]["10"][method] for f in folds],
                marker="o",label=method.title())
    ax.plot(x,[100*f["metrics"]["10"]["eclat"] for f in folds],
            color="black",linestyle="--",marker="s",linewidth=2,zorder=10,
            label="Eclat (overlaps Apriori/FP-Growth)")
    ax.set_xticks(x,[f"{int(f['origin']*100)}% training" for f in folds])
    ax.set_ylabel("Hit Rate@10 (%)")
    ax.set_title("Held-out masked-item accuracy across chronological windows")
    ax.grid(alpha=.2)
    ax.legend(ncol=2,frameon=False,loc="upper center",bbox_to_anchor=(.5,-.17))
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(output/f"figure_1_accuracy.{ext}",dpi=250,bbox_inches="tight")
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.2,4.7))
    x=np.arange(len(rows))
    w=.26
    for i,(method,label) in enumerate((("eclat","Eclat"),("apriori","Apriori"),("fpgrowth","FP-Growth"))):
        ax.bar(x+(i-1)*w,[r[f"{method}_seconds"] for r in rows],w,label=label)
    ax.set_xticks(x,[f"{int(r['origin']*100)}% / {r['support']:.2f}" for r in rows],rotation=30,ha="right")
    ax.set_ylabel("Single-run elapsed time (s)")
    ax.set_xlabel("Training prefix / minimum support")
    ax.set_title("Exploratory mining runtime on genuine UK retail invoices")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(output/f"figure_2_runtime.{ext}",dpi=250,bbox_inches="tight")
    plt.close(fig)
    accuracy=[]
    for f in folds:
        m=f["metrics"]["10"];d=m["eclat_minus_cosine"]
        lo,hi=d["cluster_bootstrap_95pct_ci"]
        accuracy.append(f"| {int(f['origin']*100)}% | {f['eligible']:,} | {m['eclat']:.2%} | {m['apriori']:.2%} | {m['fpgrowth']:.2%} | {m['cosine']:.2%} | {m['popularity']:.2%} | {100*d['difference']:+.2f} [{100*lo:+.2f}, {100*hi:+.2f}] |")
    cost=[]
    for r in rows:
        cost.append(f"| {int(r['origin']*100)}% | {r['support']:.2f} | {r['itemsets']:,} | {r['eclat_seconds']:.3f} | {r['apriori_seconds']:.3f} | {r['fpgrowth_seconds']:.3f} |")
    manuscript=f"""# Eclat, FP-Growth and Apriori for UK retail itemset mining and matched basket-completion recommendation

**Research manuscript — automatically generated from verified experimental JSON; not yet peer reviewed or submitted.**

## Abstract
**Background:** Frequent-itemset miners differ computationally, but identical itemsets and ranking policies should yield identical recommendations. **Objective:** Evaluate a vertical transaction-ID Eclat implementation against FP-Growth and Apriori on real UK invoices. **Methods:** The official UCI Online Retail dataset was cleaned using a documented UK positive-transaction policy. Three chronological training prefixes (50%, 60%, 70%) and two support levels (1%, 2%) were used for exact frequent-itemset and runtime comparisons. For recommendation evaluation, directional pair-confidence rules (1% support; 20% confidence) were evaluated on the next 10% of invoices, masking one training-known item per eligible invoice. Cosine similarity and popularity were additional baselines. Customer-clustered paired bootstrap intervals used {rec['bootstrap']} resamples. **Results:** All {len(rows)} itemset mining conditions showed exact equivalence across three algorithms. All {len(folds)} evaluation folds yielded identical Eclat, Apriori and FP-Growth predictions. Accuracy and measured runtime are reported in Tables 1 and 2. **Conclusion:** The choice among equivalent frequent-itemset mining algorithms affects computational characteristics, not ranking accuracy when itemsets and scoring are held constant. Findings do not establish online uplift.

**Keywords:** Eclat; vertical itemsets; Apriori; FP-Growth; market basket; retail; recommender systems; temporal evaluation

## 1. Introduction
Apriori [2] mines itemsets through candidate generation, FP-Growth [3] through pattern growth, and Eclat [4] through intersections of vertical transaction-ID lists. Computational performance is often conflated with recommendation quality. This study tests mining equivalence, paired recommendation equivalence and the quality of shared association-rule rankings against cosine and popularity baselines.

## 2. Data and methods
### 2.1 Source and cleaning
Official UCI Online Retail [1]. Exclude missing invoice/product/date, cancelled invoices, non-positive quantity or unit price, and transactions outside the United Kingdom. Duplicate invoice–product occurrences are collapsed to binary presence. Data are acquired directly from UCI in CI; the workbook is not redistributed.

### 2.2 Vertical Eclat implementation
The custom Eclat implementation converts each product to the set of invoice row identifiers in which it occurs. Frequent sets are extended depth-first by intersecting transaction-ID sets. A support threshold corresponds to the ceiling of support times training invoice count. Maximum itemset length is two for primary comparability with the previous projects. Each Eclat itemset and its support must match both mlxtend Apriori and FP-Growth within 1e-10; mismatches stop the workflow.

### 2.3 Chronological evaluation
Mining uses the earliest 50%, 60% and 70% of chronologically ordered invoices at support 0.01 and 0.02. For recommendation, each prefix is followed by its next non-overlapping 10% test window. One deterministic, training-known product is concealed from eligible invoices with at least two known products. The remaining items are the context. This evaluates within-invoice completion, not prediction of future orders.

### 2.4 Ranking and inference
Two-item supports become directional rules using confidence P(B|A); rules require support 0.01 and confidence at least 0.20. Confidence scores sum over observed antecedents; all three miners share candidate exclusions, popularity fallback and stable tie-breaking. Binary invoice co-occurrence cosine similarity and item popularity provide additional benchmarks. Report Hit Rate@5 and Hit Rate@10. Customer-clustered paired percentile bootstrap intervals use {rec['bootstrap']} resamples; missing customer IDs become invoice-specific clusters. These intervals are exploratory and unadjusted for multiplicity.

## 3. Results
### 3.1 Recommendation accuracy
**Table 1.** Held-out Hit Rate@10 and paired Eclat-minus-cosine difference (percentage points; 95% clustered bootstrap interval).

| Train origin | Eligible invoices | Eclat | Apriori | FP-Growth | Cosine | Popularity | Eclat − cosine (pp, 95% CI) |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(accuracy)}

The three mining-based methods produce identical predictions under the matched rule-scoring policy. This equivalence is algorithmically expected and independently asserted in the evaluation code.

### 3.2 Frequent-itemset correctness and computational cost
**Table 2.** Identical itemset counts and exploratory single-run elapsed seconds.

| Training prefix | Min support | Identical itemsets | Eclat (s) | Apriori (s) | FP-Growth (s) |
|---|---:|---:|---:|---:|---:|
{chr(10).join(cost)}

### 3.3 Figures
Figure 1: Held-out masked-item Hit Rate@10 by algorithm and chronological test window. Eclat is plotted with a dashed marker over the matching Apriori/FP-Growth line.

Figure 2: Single-run itemset mining time by training prefix and support threshold. No inferential runtime superiority is claimed.

## 4. Discussion
The main finding is the separation between frequent-itemset discovery algorithms and recommendation ranking policies. Identical frequent sets and confidence scoring generate identical ranking outputs. The additional cosine and popularity baselines show how rule-based ranking compares with other training-only association signals. Computational differences must be evaluated independently from predictive differences, ideally with repeated randomized execution order and process-level peak-memory profiling.

## 5. Limitations
Historical single-source UK retail records; only pair-itemsets in primary evaluation; masked-item completion is not next-order prediction; eligibility excludes cold-start targets; adjacent time windows are not independent external datasets; customer IDs may recur across folds; missing IDs weaken clustering; bootstrap intervals are exploratory without multiplicity correction; timings are single-run, order- and machine-dependent; tracemalloc does not measure process RSS; no revenue or causal impact can be inferred.

## 6. Conclusion
A custom vertical Eclat miner reproduced Apriori and FP-Growth itemsets in every tested condition and yielded identical pair-rule recommendations under controlled scoring. The study provides a reproducible comparative benchmark while avoiding claims of predictive advantage from an equivalent mining procedure.

## Reproducibility and data availability
Code and CI: https://github.com/AlDSChriSOB/UK-Retail-Intelligence-Apriori/tree/main/projects/project03_eclat (available after PR merge). Source data: https://archive.ics.uci.edu/dataset/352/online+retail . CI artifacts contain experimental JSON and vector/raster figures.

## Declarations pending author confirmation
Affiliation, author contributions, funding, competing interests, ethics/data governance and AI assistance must be reviewed before journal submission.

## References
1. Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
2. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules in large databases. Proceedings of VLDB, 487–499. https://www.vldb.org/conf/1994/P487.PDF
3. Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. ACM SIGMOD, 1–12. https://doi.org/10.1145/342009.335372
4. Zaki, M. J. (2000). Scalable algorithms for association mining. IEEE Transactions on Knowledge and Data Engineering, 12(3), 372–390. https://doi.org/10.1109/69.846291
5. Efron, B. (1979). Bootstrap methods: Another look at the jackknife. The Annals of Statistics, 7(1), 1–26. https://doi.org/10.1214/aos/1176344552
"""
    (output/"PROJECT03_MANUSCRIPT.md").write_text(manuscript,encoding="utf-8")
    return output

def main():
    p=argparse.ArgumentParser()
    p.add_argument("mining_json",type=Path)
    p.add_argument("recommendation_json",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project03_paper"))
    a=p.parse_args()
    print(build(a.mining_json,a.recommendation_json,a.output))
if __name__=="__main__":
    main()
