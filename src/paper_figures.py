"""Create publication figures and a tabular supplement from verified JSON evidence.

Usage: python -m src.paper_figures reports/generated/robust_benchmark.json
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def render(evidence: Path, output: Path):
    result = json.loads(evidence.read_text(encoding="utf-8"))
    folds = result["folds"]
    if len(folds) != 3 or any(len(f["configurations"]) != 9 for f in folds):
        raise ValueError("Expected three chronological windows and nine configurations per window")
    output.mkdir(parents=True, exist_ok=True)
    methods = ("cosine", "apriori", "cooccurrence", "popularity")
    fixed = []
    for fold in folds:
        matching = [c for c in fold["configurations"] if
                    np.isclose(c["min_support"], .01) and np.isclose(c["min_confidence"], .2)]
        if len(matching) != 1:
            raise ValueError("Missing fixed support/confidence configuration")
        fixed.append(matching[0])
    x = np.arange(len(folds))
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for method in methods:
        y = [100*c["metrics"]["10"][method] for c in fixed]
        ax.plot(x, y, marker="o", linewidth=2, label=method.replace("_"," ").title())
    ax.set_xticks(x, [f"{int(f['origin']*100)}% training" for f in folds])
    ax.set_ylabel("Hit Rate@10 (%)")
    ax.set_xlabel("Chronological training origin")
    ax.set_title("Held-out basket completion by method")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(output/"figure_1_method_comparison.pdf")
    fig.savefig(output/"figure_1_method_comparison.png",dpi=220)
    plt.close(fig)

    supports=sorted(set(c["min_support"] for c in folds[0]["configurations"]))
    confidences=sorted(set(c["min_confidence"] for c in folds[0]["configurations"]))
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.5),sharey=True)
    for ax,fold in zip(axes,folds):
        matrix=np.array([[100*next(c for c in fold["configurations"]
            if np.isclose(c["min_support"],s) and np.isclose(c["min_confidence"],q))
            ["metrics"]["10"]["apriori"] for q in confidences] for s in supports])
        im=ax.imshow(matrix,vmin=0,vmax=25,cmap="viridis",aspect="auto")
        ax.set_title(f"{int(fold['origin']*100)}% training")
        ax.set_xticks(range(len(confidences)),[f"{v:.2f}" for v in confidences])
        ax.set_yticks(range(len(supports)),[f"{v:.3f}" for v in supports])
        ax.set_xlabel("Minimum confidence")
        for i in range(len(supports)):
            for j in range(len(confidences)):
                ax.text(j,i,f"{matrix[i,j]:.1f}",ha="center",va="center",
                        color="white" if matrix[i,j]<12 else "black",fontsize=8)
    axes[0].set_ylabel("Minimum support")
    fig.colorbar(im,ax=axes.tolist(),label="Apriori Hit Rate@10 (%)",shrink=.78)
    fig.subplots_adjust(left=.09,right=.88,bottom=.18,top=.85,wspace=.18)
    fig.savefig(output/"figure_2_threshold_sensitivity.pdf")
    fig.savefig(output/"figure_2_threshold_sensitivity.png",dpi=220)
    plt.close(fig)

    with (output/"supplementary_grid.csv").open("w",newline="",encoding="utf-8") as handle:
        writer=csv.writer(handle)
        writer.writerow(["train_fraction","eligible","min_support","min_confidence","k",
                         "method","hit_rate","apriori_minus_method","ci_low","ci_high"])
        for fold in folds:
            for cfg in fold["configurations"]:
                for k in result["k_values"]:
                    metric=cfg["metrics"][str(k)]
                    for method in methods:
                        comparison=metric.get("apriori_minus_"+method)
                        interval=comparison["customer_cluster_95pct_ci"] if comparison else ["",""]
                        writer.writerow([fold["origin"],fold["eligible"],cfg["min_support"],
                            cfg["min_confidence"],k,method,metric[method],
                            comparison["difference"] if comparison else "",*interval])
    return output

def main():
    p=argparse.ArgumentParser()
    p.add_argument("evidence",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/paper_figures"))
    args=p.parse_args()
    print(render(args.evidence,args.output))
if __name__=="__main__":
    main()
