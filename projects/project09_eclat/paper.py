"""Evidence-only manuscript and figures for Project 09."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def build(report_path,output):
    report=json.loads(Path(report_path).read_text())
    rows=report["rows"]
    if len(rows)!=6 or any(not r["exact_itemset_equivalence"] or r["max_support_abs_delta"]>1e-10 for r in rows):
        raise AssertionError("Missing folds or miner disagreement")
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    labels=[f"{int(r['origin']*100)}% / {r['min_support']:.0%}" for r in rows]
    methods=[("eclat","Eclat"),("apriori","Apriori"),("fpgrowth","FP-Growth")]
    for key,ylabel,filename in [("seconds","Elapsed time (s)","figure_1_runtime"),("python_peak_mib","Python-tracked peak allocation (MiB)","figure_2_allocations")]:
        fig,ax=plt.subplots(figsize=(9,4.5));x=np.arange(len(rows))
        for j,(method,name) in enumerate(methods):
            ax.bar(x+(j-1)*.25,[r[f"{method}_{key}"] for r in rows],.24,label=name)
        ax.set_xticks(x,labels,rotation=25,ha="right");ax.set_ylabel(ylabel)
        ax.set_title("Three-miner comparison on identical chronological training prefixes")
        ax.legend(frameon=False);fig.tight_layout()
        for ext in ("png","pdf"):fig.savefig(output/f"{filename}.{ext}",dpi=250)
        plt.close(fig)
    table="\n".join(f"| {int(r['origin']*100)}% | {r['min_support']:.0%} | {r['train_invoices']:,} | {r['itemsets']:,} | {r['max_support_abs_delta']:.2g} | {r['eclat_seconds']:.4f} | {r['apriori_seconds']:.4f} | {r['fpgrowth_seconds']:.4f} | {r['eclat_python_peak_mib']:.2f} | {r['apriori_python_peak_mib']:.2f} | {r['fpgrowth_python_peak_mib']:.2f} |" for r in rows)
    manuscript=f"""# Exact Eclat, Apriori and FP-Growth frequent-itemset equivalence on UK retail transactions

**Research status:** Evidence-generated draft; not peer-reviewed or submitted.

## Abstract
**Objective:** Validate a transparent vertical Eclat implementation against Apriori and FP-Growth on genuine retail invoices. **Methods:** The official UCI Online Retail dataset [1] was cleaned to positive UK invoices. Six train-only conditions combined three chronological training prefixes (50%, 60%, 70%) with two minimum supports (1%, 2%). Frequent itemsets up to length two were compared exactly. **Results:** All six configurations passed the strict itemset-identity and support-equivalence gates; measured runtimes and Python-tracked allocations are reported in Table 1 and Figures 1–2. **Conclusion:** Exactness is demonstrated under the tested configurations, while exploratory timings cannot establish platform-independent performance superiority.

## 1. Introduction
Frequent itemset mining supports association analysis of transaction baskets. Apriori [2] uses candidate enumeration, FP-Growth [3] uses compressed pattern-growth, and Eclat [4] uses vertical transaction-ID intersections. An independent reference implementation provides an opportunity to check exact agreement on real transactions.

## 2. Methods
Official UCI Online Retail [1] was filtered to positive, non-cancelled UK invoices. Invoice-product presence was encoded as a sparse binary matrix and sorted chronologically. Eclat constructs per-item transaction-ID sets and recursively intersects them; minimum frequency uses ceil(minimum_support × number_of_training_invoices). Apriori and FP-Growth were called from mlxtend with the same support and maximum itemset length. All three algorithms were run on identical training prefixes. Equality of itemset keys and absolute support deviation at most 1e-10 was mandatory. Wall-clock timing used a sequential single-run timer; tracemalloc measured Python-tracked allocations rather than full process memory.

## 3. Results
**Table 1.** Exactness and exploratory measurements from the actual workflow.

| Train prefix | Support | Invoices | Itemsets | Max support delta | Eclat s | Apriori s | FP-Growth s | Eclat MiB | Apriori MiB | FP-Growth MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

**Figure 1:** Runtime by chronological prefix and minimum support.

**Figure 2:** Python-tracked peak allocation by chronological prefix and minimum support.

## 4. Discussion
The exact itemset and support agreement provides a direct correctness check for the vertical Eclat implementation under the tested settings. Timing differences should not be generalized: Python set intersections and library miners have different implementation overheads, and sequential single runs are susceptible to system noise.

## 5. Limitations
One historical retailer; only two support thresholds and pairwise itemsets; no independent replication, repeated timing distributions or process-RSS profiling; no recommendation-accuracy or online business evaluation.

## 6. Conclusion
This benchmark validates exact agreement of Eclat, Apriori and FP-Growth on six real-data training conditions, and reports descriptive efficiency measurements suitable for follow-up robustness research.

## Reproducibility and declarations
Evidence JSON and figures were generated by the repository's GitHub Actions workflow using the original UCI workbook. Confirm authorship, affiliations, funding, conflicts and AI-assistance declarations before submission.

## References
1. Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
2. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. VLDB.
3. Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. ACM SIGMOD. https://doi.org/10.1145/342009.335372
4. Zaki, M. J. (2000). Scalable algorithms for association mining. IEEE Transactions on Knowledge and Data Engineering, 12(3), 372–390. https://doi.org/10.1109/69.846291
"""
    (output/"PROJECT09_MANUSCRIPT.md").write_text(manuscript,encoding="utf-8")

def main():
    p=argparse.ArgumentParser();p.add_argument("report",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project09_paper"))
    a=p.parse_args();build(a.report,a.output)
if __name__=="__main__":main()
