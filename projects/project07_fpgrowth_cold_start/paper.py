"""Generate Project 07 figures and a manuscript only from measured JSON."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def build(report_path,output):
    report=json.loads(Path(report_path).read_text())
    folds=report["folds"]
    methods=report["methods"];strata=report["strata"]
    if len(folds)!=3 or any(set(f["strata"])!=set(strata) for f in folds):
        raise AssertionError("Incomplete experimental results")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    counts={s:sum(f["strata"][s]["eligible"] for f in folds) for s in strata}
    scores={}
    for s in strata:
        for m in methods:
            valid=[f for f in folds if f["strata"][s]["eligible"]]
            scores[s,m]=sum(f["strata"][s]["eligible"]*f["strata"][s]["metrics"][m]["hit_rate_10"]
                            for f in valid)/counts[s] if counts[s] else float("nan")
    x=np.arange(len(strata))
    fig,ax=plt.subplots(figsize=(8.4,5))
    for j,m in enumerate(methods):
        ax.bar(x+(j-1)*.25,[100*scores[s,m] for s in strata],.24,label=m)
    ax.set_xticks(x,[f"{s}\n(n={counts[s]:,})" for s in strata])
    ax.set_ylabel("Pooled held-out Hit Rate@10 (%)")
    ax.set_title("Recommendation accuracy by training-only item-frequency stratum")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(output/f"figure_1_stratified_accuracy.{ext}",dpi=250)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.6))
    width=.23
    for j,f in enumerate(folds):
        ax.bar(x+(j-1)*width,[f["strata"][s]["eligible"] for s in strata],
               width,label=f"{int(f['origin']*100)}% training prefix")
    ax.set_xticks(x,strata)
    ax.set_ylabel("Eligible held-out target invoices")
    ax.set_title("Evaluation support per frequency stratum")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(output/f"figure_2_stratum_sample_sizes.{ext}",dpi=250)
    plt.close(fig)
    rows=[]
    for f in folds:
        for s in strata:
            for m in methods:
                v=f["strata"][s]
                z=v["metrics"][m]
                fmt=lambda x: "NA" if x is None else f"{x:.2%}"
                rows.append(f"| {int(f['origin']*100)}% | {s} | {m} | {v['eligible']:,} | {fmt(z['hit_rate_5'])} | {fmt(z['hit_rate_10'])} |")
    manuscript=f"""# Cold-start and product-frequency robustness of FP-Growth retail basket completion

**Evidence-generated research draft; not peer reviewed or submitted.**

## Abstract
**Objective:** Measure whether FP-Growth association-rule recommendations exhibit uneven performance across product-frequency strata. **Methods:** Official UCI Online Retail UK positive invoices were chronologically split at training prefixes 50%, 60%, and 70%, with each subsequent 10% held out. Each eligible multi-item test basket contributed one deterministically concealed product. Product strata (training-unseen, rare, middle, popular) were derived exclusively from training invoice frequencies. FP-Growth directional pair-confidence scoring was compared with cosine co-occurrence and item popularity using the same concealed targets. **Results:** Table 1 reports per-fold denominators and Hit Rate@5/10 for all methods and strata; Figures 1–2 show pooled accuracy and stratum sizes. **Conclusion:** Frequency-stratified evaluation reveals which product segments are underserved by transaction-only recommenders; observed differences are descriptive, not causal.

## 1. Introduction
Frequent-itemset methods can be biased toward high-frequency products and are unable to infer meaningful co-occurrences for products not present in training. Aggregate recommendation accuracy may conceal this distributional weakness. This study evaluates frequency-stratified performance on a genuine retail dataset.

## 2. Methods
The UCI Online Retail dataset [1] was filtered to UK invoices with positive quantities and prices, excluding cancellations. Invoice-product occurrences were converted to binary presence and sorted chronologically. Expanding training prefixes of 50%, 60%, and 70% were each followed by a 10% held-out test window.

Each eligible held-out invoice with at least two products contributed a masked target selected using a fixed fold-specific seed. Product frequency was computed from training only. A zero-frequency product is **unseen**; positive-frequency products are partitioned by train-only 1/3 and 2/3 quantiles into rare, middle and popular strata. Quantile ties may create unequal stratum sizes or empty strata. The same targets and contexts were evaluated for all methods.

FP-Growth [2] mined length-two itemsets at 1% support and produced directional rules at 20% confidence. The shared repository ranking implementation evaluated FP-Growth rules, cosine co-occurrence, and item-popularity baselines. The task is within-invoice masked-item completion, not forecasting future orders. Per-stratum sample sizes are reported. No significance or fairness claim is made.

## 3. Results
**Table 1.** Per-fold frequency-stratified held-out performance.

| Training prefix | Target stratum | Method | Eligible | HR@5 | HR@10 |
|---|---|---|---:|---:|---:|
{chr(10).join(rows)}

**Figure 1.** Invoice-weighted pooled HR@10 across the three chronological windows by training-only target-frequency stratum. Pooled estimates are descriptive; windows are not independent.

**Figure 2.** Eligible held-out invoice counts by target stratum and fold.

## 4. Discussion
The stratified evaluation makes visible the performance of transaction-only recommenders for low-frequency and training-unseen products, while retaining common evaluation targets for matched methods. Because some product strata are small, sample sizes and fold-specific outcomes must accompany aggregate metrics. Cold-start methods using content or metadata would be a natural next experiment.

## 5. Limitations
Single historical UK retailer; within-invoice masking rather than future purchase; strata based on invoice counts rather than units or revenue; ties can yield unbalanced tertiles; no uncertainty intervals or multiplicity adjustment; adjacent chronological test windows are not independent; no demographic fairness or causal uplift inference.

## 6. Conclusion
Training-only frequency stratification provides an auditable way to test recommendation robustness beyond aggregate accuracy. Further work should compare cold-start-capable content or hybrid models on a separate untouched temporal evaluation.

## Data and reproducibility
Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
Repository: https://github.com/AlDSChriSOB/UK-Retail-Intelligence-Apriori/tree/main/projects/project07_fpgrowth_cold_start (after merge).

## Declarations
Confirm authorship, affiliation, funding, conflicts and AI assistance before submission.

## References
1. Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
2. Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. ACM SIGMOD. https://doi.org/10.1145/342009.335372
3. Schein, A. I., Popescul, A., Ungar, L. H., & Pennock, D. M. (2002). Methods and metrics for cold-start recommendations. SIGIR. https://doi.org/10.1145/564376.564421
"""
    (output/"PROJECT07_MANUSCRIPT.md").write_text(manuscript,encoding="utf-8")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("report",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project07_paper"))
    a=p.parse_args();build(a.report,a.output)
if __name__=="__main__":main()
