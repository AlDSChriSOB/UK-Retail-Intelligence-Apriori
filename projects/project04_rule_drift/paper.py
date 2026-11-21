"""Project 04: figures and manuscript generated exclusively from verified drift JSON."""
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
    if len(folds)!=3 or not all(0<=f["test_confidence_retention_rate"]<=1 for f in folds):
        raise AssertionError("Invalid evidence")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    x=np.arange(len(folds))
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    ax.bar(x-.17,[100*f["test_confidence_retention_rate"] for f in folds],.34,label="Confidence threshold")
    ax.bar(x+.17,[100*f["test_support_and_confidence_retention_rate"] for f in folds],.34,label="Support + confidence")
    ax.set_ylim(0,100)
    ax.set_xticks(x,[f"{int(f['origin']*100)}% training" for f in folds])
    ax.set_ylabel("Retained training rules among test-observed antecedents (%)")
    ax.set_title("Out-of-time association-rule threshold stability")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(output/f"figure_1_rule_retention.{ext}",dpi=250)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    means=np.asarray([f["drift"]["mean_change"] for f in folds])
    lows=np.asarray([f["drift"]["mean_change_rule_bootstrap_95pct_ci"][0] for f in folds])
    highs=np.asarray([f["drift"]["mean_change_rule_bootstrap_95pct_ci"][1] for f in folds])
    ax.errorbar(x,100*means,yerr=np.vstack((100*(means-lows),100*(highs-means))),
                fmt="o",capsize=5,color="black")
    ax.axhline(0,linestyle="--",linewidth=1,color="gray")
    ax.set_xticks(x,[f"{int(f['origin']*100)}% training" for f in folds])
    ax.set_ylabel("Mean held-out minus training confidence (pp)")
    ax.set_title("Descriptive rule-level bootstrap intervals")
    fig.tight_layout()
    for ext in ("png","pdf"):
        fig.savefig(output/f"figure_2_confidence_drift.{ext}",dpi=250)
    plt.close(fig)
    rows=[]
    for f in folds:
        d=f["drift"];lo,hi=d["mean_change_rule_bootstrap_95pct_ci"]
        rows.append(f"| {int(f['origin']*100)}% | {f['train_rules']:,} | {f['observed_antecedent_rules']:,} | {f['test_confidence_retention_rate']:.2%} | {f['test_support_and_confidence_retention_rate']:.2%} | {100*d['mean_change']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    paper=f"""# Out-of-time stability of UK retail association rules: a chronological replication study

**Status:** Research manuscript generated from verified experiment JSON; not submitted or peer reviewed.

## Abstract
**Objective:** Assess whether association rules mined from historical UK retail transactions retain confidence and support in subsequent invoice windows. **Methods:** Genuine UCI Online Retail transactions were cleaned to positive non-cancelled UK invoices. Directional two-product rules were selected exclusively from chronological training prefixes of 50%, 60% and 70% at minimum support 1% and confidence 20%. Each rule was re-evaluated on the following 10% of invoices. We report test-antecedent observability, threshold retention and mean paired confidence changes with {report['bootstrap']} rule-level bootstrap replicates. **Results:** Table 1 presents directly measured fold-specific retention and confidence changes. **Conclusion:** This retrospective study quantifies rule stability; descriptive intervals do not establish independent rule-level statistical significance, causality or online recommendation effectiveness.

## 1. Introduction
Association-rule models are typically evaluated for in-sample support, confidence and lift [2]. For operational decision-making, a rule must also remain useful as purchasing patterns change. A chronological holdout makes this distinction observable without using future invoices for training-rule selection.

## 2. Dataset and methods
### 2.1 Source
Official UCI Online Retail [1], using only UK invoices after excluding missing invoice/product/date, cancelled invoices, nonpositive quantity and nonpositive price. Repeated invoice–product rows are converted to binary presence.

### 2.2 Rule discovery
Directional pairs are mined using training-only co-occurrence counts. The minimum support is 0.01 of training invoices; minimum directional confidence is 0.20. Prefixes at 50%, 60%, 70% of chronological invoices are followed by their respective 10% test windows. Test periods do not inform rule selection.

### 2.3 Out-of-time replication
For each selected A→B rule, test confidence is the fraction of test invoices containing A that also contain B. Rules with no test A occurrences are excluded from conditional confidence comparisons and counted separately. Retention is reported both for confidence alone and for joint test support and confidence thresholds. Rule-level mean confidence changes are paired within each training-selected rule.

### 2.4 Descriptive uncertainty
For each fold, {report['bootstrap']} bootstrap samples resample selected rules with replacement, preserving train/test pairing. Since rules share products and invoice histories, rule observations are correlated: these percentile intervals are descriptive and must not be treated as independent-sampling hypothesis tests. No multiple-comparison correction or causal interpretation is claimed.

## 3. Results
**Table 1.** Out-of-time rule retention and paired confidence drift (percentage points).

| Training origin | Training rules | Test-observed antecedents | Confidence retained | Joint thresholds retained | Mean confidence change (pp) | Rule-bootstrap 95% interval (pp) |
|---|---:|---:|---:|---:|---:|---|
{chr(10).join(rows)}

**Figure 1.** Test-window confidence and joint-threshold retention for training-selected rules.

**Figure 2.** Mean out-of-time confidence change with descriptive rule-level bootstrap intervals.

## 4. Discussion
Training confidence need not generalize to subsequent purchasing periods. Threshold retention provides a directly interpretable measure of replication under fixed decision rules, while mean confidence change quantifies directional change among rules with observable test antecedents. The study intentionally does not infer customer behaviour causes or commercial lift.

## 5. Limitations
Single historical UK dataset; nonindependent rules; no customer-level clustered inference; bootstrap intervals are descriptive; missing test antecedents create conditional estimand selection; three adjacent chronological test windows are not external replications; only pair rules at fixed thresholds; no next-order recommendation evaluation, online intervention or revenue uplift.

## 6. Conclusion
The reproducible protocol tests temporal rule stability without future-information leakage in rule discovery. Interpretation must distinguish descriptive rule replication from causal drift or predictive recommendation claims.

## Data and code availability
Official UCI Online Retail: https://archive.ics.uci.edu/dataset/352/online+retail . Source code and CI in https://github.com/AlDSChriSOB/UK-Retail-Intelligence-Apriori/tree/main/projects/project04_rule_drift (after merge). Figures and numerical evidence are retained as GitHub Actions artifacts.

## Author declarations to confirm before submission
Affiliation, funding, conflicts of interest, author contributions, ethics/data governance and AI-assistance declarations.

## References
1. Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
2. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules in large databases. Proceedings of VLDB, 487–499. https://www.vldb.org/conf/1994/P487.PDF
3. Efron, B. (1979). Bootstrap methods: Another look at the jackknife. The Annals of Statistics, 7(1), 1–26. https://doi.org/10.1214/aos/1176344552
"""
    (output/"PROJECT04_MANUSCRIPT.md").write_text(paper,encoding="utf-8")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("report",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project04_paper"))
    a=p.parse_args()
    build(a.report,a.output)
if __name__=="__main__":
    main()
