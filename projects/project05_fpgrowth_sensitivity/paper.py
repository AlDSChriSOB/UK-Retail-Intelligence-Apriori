"""Project 05: evidence-driven sensitivity heatmap and manuscript."""
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
    if len(folds)!=3 or any(len(f["grid"])!=9 for f in folds):
        raise AssertionError("Expected 3 folds and nine fixed settings per fold")
    supports=sorted({r["support"] for r in folds[0]["grid"]})
    confidences=sorted({r["confidence"] for r in folds[0]["grid"]})
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    avg=np.asarray([[np.mean([next(r["hit_rate_10"] for r in f["grid"]
                                   if r["support"]==s and r["confidence"]==c)
                              for f in folds])
                     for c in confidences] for s in supports])
    fig,ax=plt.subplots(figsize=(7,5))
    im=ax.imshow(avg*100,aspect="auto",cmap="viridis")
    ax.set_xticks(range(len(confidences)),[f"{c:.0%}" for c in confidences])
    ax.set_yticks(range(len(supports)),[f"{s:.1%}" for s in supports])
    ax.set_xlabel("Minimum directional confidence")
    ax.set_ylabel("Minimum itemset support")
    ax.set_title("FP-Growth HR@10: mean across three chronological windows")
    for i in range(len(supports)):
        for j in range(len(confidences)):
            ax.text(j,i,f"{100*avg[i,j]:.2f}%",ha="center",va="center",
                    color="white" if avg[i,j]<np.mean(avg) else "black")
    fig.colorbar(im,ax=ax,label="Mean Hit Rate@10 (%)")
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(output/f"figure_1_accuracy_sensitivity.{ext}",dpi=250)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.5))
    for c in confidences:
        y=[np.mean([next(r["directional_rules"] for r in f["grid"]
                         if r["support"]==s and r["confidence"]==c) for f in folds])
           for s in supports]
        ax.plot([100*s for s in supports],y,marker="o",label=f"Confidence {c:.0%}")
    ax.set_xlabel("Minimum support (%)")
    ax.set_ylabel("Mean directional rule count")
    ax.set_title("Model complexity across mining thresholds")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(output/f"figure_2_rule_complexity.{ext}",dpi=250)
    plt.close(fig)
    lines=[]
    for f in folds:
        for r in f["grid"]:
            lines.append(f"| {int(f['origin']*100)}% | {r['support']:.1%} | {r['confidence']:.0%} | {r['itemsets']:,} | {r['directional_rules']:,} | {r['hit_rate_5']:.2%} | {r['hit_rate_10']:.2%} |")
    manuscript=f"""# Support–confidence sensitivity of FP-Growth retail recommendations under chronological evaluation

**Research manuscript — generated from actual UCI experimental results; not peer reviewed or submitted.**

## Abstract
**Objective:** Quantify how minimum itemset support and directional rule confidence change model complexity and held-out within-invoice recovery. **Methods:** Genuine UCI Online Retail transactions were cleaned to positive UK invoices. FP-Growth mined singleton and pair itemsets on chronological 50%, 60% and 70% training prefixes. A prespecified 3×3 grid of support (0.5%, 1%, 2%) and confidence (10%, 20%, 40%) was evaluated on the following 10% invoice window per prefix. One deterministic training-known item was masked from each eligible test basket. **Results:** All 27 configurations and their HR@5/10, frequent itemsets and rule counts are reported in Table 1. **Conclusion:** Threshold sensitivity must be assessed independently of algorithmic correctness. These results are descriptive; no configuration is selected on test performance for a claim of out-of-sample optimized accuracy.

## 1. Introduction
Frequent-pattern mining [2,3] depends on support and confidence thresholds. High thresholds may yield simpler rule models but miss less common product associations. Low thresholds can increase complexity and may alter ranking quality. This study measures both effects on real transactions using a fixed, auditable grid.

## 2. Data and methods
The official UCI Online Retail dataset [1] was filtered to UK positive, non-cancelled invoices, retaining missing customer IDs for invoice-level analysis. Repeated invoice-product occurrences were collapsed to binary presence. Chronological training prefixes at 50%, 60%, 70% are followed by their next 10% held-out windows.

FP-Growth mines itemsets of length at most two at support thresholds 0.005, 0.01, 0.02. Each itemset set is converted to directional pair-confidence rules at thresholds 0.10, 0.20, 0.40. Candidate ranking uses the shared deterministic rule-scoring and popularity fallback. A training-known product is masked per eligible invoice using a fixed random seed independent of configuration. Report Hit Rate@5 and Hit Rate@10. No test-window optimization or significance claims are made.

## 3. Results
**Table 1.** All 27 prespecified fold–threshold configurations.

| Training prefix | Support | Confidence | Frequent itemsets | Directional rules | HR@5 | HR@10 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

**Figure 1.** Mean HR@10 across three chronological test windows by fixed support and confidence thresholds.

**Figure 2.** Mean number of directional rules by support and confidence threshold.

## 4. Discussion
The study maps the sensitivity of the rule-based recommender to mining thresholds, without treating the highest test-window result as a selected deployable model. In a deployment-oriented extension, hyperparameter selection would require a nested chronological validation window and a separate untouched final test period.

## 5. Limitations
Single historical dataset; within-invoice masked-item recovery rather than future purchasing; three adjacent windows not independent external replications; fixed thresholds and pair-only itemsets; no customer-clustered uncertainty or multiple-comparison inference; observed differences are descriptive; no online conversion or causal uplift claim.

## 6. Conclusion
Prespecified threshold grids reveal the tradeoff between rule count and masked-item recovery in FP-Growth-based recommendations. A reproducible sensitivity analysis is a prerequisite, not a substitute, for nested temporal model selection.

## Data and code
UCI Online Retail: https://archive.ics.uci.edu/dataset/352/online+retail . Reproducible code and CI: https://github.com/AlDSChriSOB/UK-Retail-Intelligence-Apriori/tree/main/projects/project05_fpgrowth_sensitivity (after merge).

## Declarations
Author affiliation, contributions, funding, conflicts, data governance and AI assistance require confirmation before journal submission.

## References
1. Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
2. Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. ACM SIGMOD, 1–12. https://doi.org/10.1145/342009.335372
3. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules in large databases. VLDB, 487–499. https://www.vldb.org/conf/1994/P487.PDF
"""
    (output/"PROJECT05_MANUSCRIPT.md").write_text(manuscript,encoding="utf-8")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("report",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project05_paper"))
    a=p.parse_args()
    build(a.report,a.output)
if __name__=="__main__":
    main()
