"""Generate Project 08 evidence-based research manuscript and plots."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def build(report_path,output):
    report=json.loads(Path(report_path).read_text())
    folds=report["folds"];segments=report["segments"];policies=report["policies"]
    if len(folds)!=3 or any(set(f["strata"])!=set(segments) for f in folds):
        raise AssertionError("Incomplete evidence")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    counts={s:sum(f["strata"][s]["eligible"] for f in folds) for s in segments}
    pooled={}
    for s in segments:
        for p in policies:
            pooled[s,p]=(sum(f["strata"][s]["eligible"]*f["strata"][s]["policies"][p]["hit_rate_10"]
                        for f in folds if f["strata"][s]["eligible"])/counts[s]) if counts[s] else np.nan
    x=np.arange(len(segments))
    fig,ax=plt.subplots(figsize=(9,5))
    for j,p in enumerate(policies):
        ax.bar(x+(j-1)*.25,[100*pooled[s,p] for s in segments],.24,label=p.replace("_"," "))
    ax.set_xticks(x,[f"{s}\n(n={counts[s]:,})" for s in segments])
    ax.set_ylabel("Invoice-weighted Hit Rate@10 (%)")
    ax.set_title("Global versus customer-segment FP-Growth recommendation")
    ax.legend(frameon=False,fontsize=8)
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(output/f"figure_1_segment_accuracy.{ext}",dpi=250)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5))
    for j,f in enumerate(folds):
        ax.bar(x+(j-1)*.23,[f["strata"][s]["eligible"] for s in segments],.23,
               label=f"{int(f['origin']*100)}% training prefix")
    ax.set_xticks(x,segments)
    ax.set_ylabel("Eligible held-out invoices")
    ax.set_title("RFM-segment evaluation coverage")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png","pdf"):fig.savefig(output/f"figure_2_segment_counts.{ext}",dpi=250)
    plt.close(fig)
    rows=[]
    for f in folds:
        for s in segments:
            for p in policies:
                d=f["strata"][s];m=d["policies"][p]
                fmt=lambda v:"NA" if v is None else f"{v:.2%}"
                rows.append(f"| {int(f['origin']*100)}% | {s} | {p} | {d['eligible']:,} | {fmt(m['hit_rate_5'])} | {fmt(m['hit_rate_10'])} |")
    manuscript=f"""# Does customer-level RFM segmentation improve FP-Growth basket completion? A chronological UK retail evaluation

**Status:** Evidence-generated draft; not peer-reviewed or submitted.

## Abstract
**Objective:** Compare customer-segment FP-Growth with global FP-Growth and popularity ranking. **Methods:** On cleaned positive UK invoices from the genuine UCI Online Retail dataset [1], customer RFM features and tertile segments were fitted exclusively to chronological training prefixes of 50%, 60%, and 70%. Each following 10% of invoices was evaluated with identical masked-item trials, using train-known targets. **Results:** Table 1 reports observed segment-specific Hit Rate@5/10 and sample sizes for all three policies; Figures 1 and 2 show pooled descriptive accuracy and sample sizes. **Conclusion:** The experiment quantifies the accuracy and coverage tradeoff of segment-specific association rules without claiming causal uplift or independently validated optimality.

## 1. Introduction
Retail customers differ in purchase frequency, recency and spend. Segment-specific association rules might reflect behavioural heterogeneity more closely than global rules, but partitioning reduces the number of training invoices. This study measures that tradeoff with chronological evaluation.

## 2. Methods
Official UCI Online Retail data [1] were filtered to positive non-cancelled UK invoices. In each chronological fold, customer RFM was calculated from training invoices only: recency in days since last training purchase, frequency as unique invoice count, and monetary value as summed positive line revenue. Within-training percentile ranks were added to form a score and partition customers into low, middle and high segments by training-score tertiles. Missing or previously unseen CustomerID receives the global-model fallback.

FP-Growth [2] mined up to two-item patterns at minimum support 1% and directional confidence 20%, separately on global training invoices and each training customer segment. A global item-popularity ranking provides a baseline. Identical eligible held-out invoices and randomly masked training-known target items were scored for all policies, using a fixed seed. Hit Rate@5 and Hit Rate@10 were calculated within each segment and fold.

This is within-invoice item completion, not prediction of a customer's next purchase. All comparisons are descriptive; no significance tests or revenue uplift are claimed.

## 3. Results
**Table 1.** Observed held-out accuracy with denominators.

| Training prefix | Training-derived customer segment | Policy | Eligible | HR@5 | HR@10 |
|---|---|---|---:|---:|---:|
{chr(10).join(rows)}

**Figure 1.** Invoice-weighted pooled HR@10 by segment and policy across the three evaluation windows.

**Figure 2.** Number of eligible masked-item invoices by training-derived customer segment and chronological fold.

## 4. Discussion
Segment-specific rule mining can increase behavioural specificity while reducing the training support for individual item pairs. Global fallback provides defined behaviour for previously unseen or unidentified customers. Because the three chronological test windows are neighbouring periods, their outcomes are not independent external replications. Segment-level comparisons should be interpreted alongside their denominators.

## 5. Limitations
Single historical retailer, RFM-derived behavioural rather than demographic groups, segment-specific support denominators, potentially small subgroup samples, invoice-level masking instead of future purchasing, no held-out policy selection, no causal lift or online experiment, and no statistical significance claims.

## 6. Conclusion
A reproducible, training-only segmentation pipeline enables a fair descriptive comparison of global and customer-segment FP-Growth recommendations. Future studies should use untouched later periods and customer-level future-order prediction.

## Reproducibility and declarations
Data: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
Code: https://github.com/AlDSChriSOB/UK-Retail-Intelligence-Apriori/tree/main/projects/project08_fpgrowth_segments (after merge).
Confirm authorship, affiliations, funding, conflicts, and AI assistance before submission.

## References
1. Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
2. Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. ACM SIGMOD. https://doi.org/10.1145/342009.335372
3. Hughes, A. M. (1994). Strategic Database Marketing. Probus Publishing.
"""
    (output/"PROJECT08_MANUSCRIPT.md").write_text(manuscript,encoding="utf-8")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("report",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project08_paper"))
    a=p.parse_args();build(a.report,a.output)
if __name__=="__main__":main()
