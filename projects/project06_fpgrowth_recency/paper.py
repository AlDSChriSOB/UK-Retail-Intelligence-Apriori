"""Evidence-only manuscript and figures for Project 06."""
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
    names=("expanding","recent_40pct","recent_20pct")
    if len(folds)!=3 or any(set(f["windows"])!=set(names) for f in folds):
        raise AssertionError("Incomplete real-data evidence")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    x=np.arange(len(folds))
    fig,ax=plt.subplots(figsize=(8,4.7))
    for j,name in enumerate(names):
        y=[100*f["windows"][name]["hit_rate_10"] for f in folds]
        ax.bar(x+(j-1)*.25,y,.24,label=name.replace("_"," "))
    ax.set_xticks(x,[f"{int(f['origin']*100)}% prefix" for f in folds])
    ax.set_ylabel("Masked-item Hit Rate@10 (%)")
    ax.set_title("FP-Growth: expanding versus recent training windows")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(output/f"figure_1_window_accuracy.{ext}",dpi=250)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.7))
    for name in names:
        ax.plot(x,[f["windows"][name]["directional_rules"] for f in folds],
                marker="o",label=name.replace("_"," "))
    ax.set_xticks(x,[f"{int(f['origin']*100)}% prefix" for f in folds])
    ax.set_ylabel("Directional association rules")
    ax.set_title("Rule complexity by training-history policy")
    ax.grid(alpha=.2);ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(output/f"figure_2_window_rule_counts.{ext}",dpi=250)
    plt.close(fig)
    rows=[]
    for f in folds:
        for name in names:
            v=f["windows"][name]
            rows.append(f"| {int(f['origin']*100)}% | {name} | {v['training_invoices']:,} | {v['directional_rules']:,} | {v['cold_target_rate']:.2%} | {v['hit_rate_5']:.2%} | {v['hit_rate_10']:.2%} |")
    deltas=[]
    for f in folds:
        for name in names[1:]:
            for k in (5,10):
                d=f["paired_comparisons"][name+"_minus_expanding"][f"hit_rate_{k}"]
                lo,hi=d["customer_cluster_95pct_ci"]
                deltas.append(f"| {int(f['origin']*100)}% | {name} | {k} | {100*d['difference']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    text=f"""# Does recent training history improve FP-Growth basket completion? A UK retail chronological study

**Status:** Evidence-generated research draft; not peer reviewed or submitted.

## Abstract
**Objective:** Compare fixed recent-history FP-Growth policies against expanding-history FP-Growth on real UK retail invoices. **Methods:** The official UCI Online Retail dataset [1] was filtered to positive, non-cancelled UK invoices. At 50%, 60%, and 70% chronological training prefixes, three prespecified policies (all history, most recent 40% of the prefix, and most recent 20%) mined two-item frequent patterns at 1% support and 20% directional confidence. Each policy was evaluated on the same masked-item trials in the subsequent 10% of invoices. Hit Rate@5/10, training-cold target rates, rule counts, and paired customer-cluster bootstrap differences were recorded. **Results:** Tables 1 and 2 contain directly observed fold-specific metrics and descriptive intervals. **Conclusion:** Recent-history truncation trades historical coverage for responsiveness, but no single policy is selected from held-out test performance.

## 1. Introduction
Frequent-pattern recommendations are sensitive to historical purchasing patterns and the volume of retained transactions. Truncating the training history can emphasize recent co-occurrences while reducing data and product coverage. This study examines that tradeoff without future-information leakage.

## 2. Methods
The official UCI Online Retail dataset [1] was cleaned to positive, non-cancelled UK invoices. Transactions were collapsed to binary invoice–product baskets and ordered by invoice timestamp. For each 50%, 60%, and 70% prefix, FP-Growth [2] mines two-item patterns from all available historical invoices or the most recent 40% or 20% of the training prefix. Minimum support and confidence are fixed at 1% and 20%. The next 10% of invoices is held out.

One item observed in the expanding training history is randomly masked from each eligible test invoice using a fixed seed; exactly the same targets and contexts are used for all policies. Some targets may be absent from recent-window training; the cold-target rate is reported. Hit Rate@5/10 is evaluated using the repository's shared deterministic directional-rule scoring and fallback.

Paired performance differences use {report['bootstrap']} customer-cluster bootstrap replicates within each fold. Intervals are descriptive, unadjusted for multiple comparisons, and should not be interpreted as independent external replications.

## 3. Results
**Table 1.** All nine prespecified history-policy evaluations.

| Training prefix | Window | Training invoices | Rules | Training-cold targets | HR@5 | HR@10 |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

**Table 2.** Recent minus expanding-history hit-rate differences (percentage points), with descriptive paired customer-cluster bootstrap intervals.

| Training prefix | Recent window | K | Difference (pp) | 95% interval (pp) |
|---|---|---:|---:|---|
{chr(10).join(deltas)}

**Figure 1.** HR@10 by chronological training origin and history policy.

**Figure 2.** Directional rule count by chronological training origin and history policy.

## 4. Discussion
The analysis compares model freshness, rule-set complexity and cold-target coverage under a matched test protocol. The observed outcomes should be interpreted as within-invoice item completion, not as future-order recommendations. Selecting a history policy based on these held-out comparisons would require a further untouched temporal test set.

## 5. Limitations
Single historical dataset; invoice-count windows rather than equal-duration time windows; pair-only rules; adjacent test windows not independent; within-invoice masking instead of future purchases; cluster-bootstrap uncertainty descriptive and unadjusted for multiple comparisons; no causal or commercial uplift inference.

## 6. Conclusion
The prespecified chronological comparison provides reproducible evidence on the sensitivity of FP-Growth basket completion to history length without claiming a test-optimized policy.

## Data and reproducibility
Chen, D. (2015), Online Retail [Dataset], UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33 . Project source and CI: https://github.com/AlDSChriSOB/UK-Retail-Intelligence-Apriori/tree/main/projects/project06_fpgrowth_recency (after merge).

## Declarations to complete before submission
Confirm authorship, affiliations, funding, conflicts, data governance and AI-assistance statements.

## References
1. Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
2. Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. ACM SIGMOD. https://doi.org/10.1145/342009.335372
3. Efron, B. (1979). Bootstrap methods: Another look at the jackknife. The Annals of Statistics, 7(1), 1–26. https://doi.org/10.1214/aos/1176344552
"""
    (output/"PROJECT06_MANUSCRIPT.md").write_text(text,encoding="utf-8")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("report",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project06_paper"))
    a=p.parse_args()
    build(a.report,a.output)
if __name__=="__main__":main()
