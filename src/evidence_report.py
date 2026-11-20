"""Generate evidence-derived manuscript-ready tables and vector figures.

Never substitutes fabricated results; reads JSON produced by heldout_topk.py.
"""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate(source, output_dir):
    data = json.loads(Path(source).read_text(encoding="utf-8"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = data["results"]
    ks = data["k_values"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for method, label in (("rules", "Association rules"), ("popularity", "Popularity"), ("random", "Random")):
        values = [100 * results[method][str(k)]["hit_rate_at_k"] for k in ks]
        ax.plot(ks, values, marker="o", linewidth=2, label=label)
    ax.set(xlabel="Recommendation list length K", ylabel="Hit rate (%)",
           title="Chronological held-out UK retail basket completion")
    ax.set_xticks(ks)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "hit_rate_at_k.pdf")
    fig.savefig(output_dir / "hit_rate_at_k.png", dpi=300)
    plt.close(fig)

    rows = []
    for method in ("rules", "popularity", "random"):
        for k in ks:
            r = results[method][str(k)]
            ci = r["hit_rate_95pct_percentile_ci"]
            rows.append(f'| {method} | {k} | {100*r["hit_rate_at_k"]:.3f} | [{100*ci[0]:.3f}, {100*ci[1]:.3f}] | {100*r["precision_at_k"]:.3f} | {100*r["catalogue_coverage_at_k"]:.3f} |')
    report = f"""# Chronological basket-completion evaluation: executed evidence

## Research question
Can train-only directional association rules improve recovery of one concealed product from later UK retail invoices, relative to train-only popularity and seeded random ranking?

## Source and protocol
Official UCI Online Retail, UK positive non-cancelled transactions. {data["train_invoices"]:,} chronological training invoices and {data["test_invoices"]:,} later test invoices. {data["eligible_test_invoices"]:,} eligible test invoices, training catalogue {data["train_catalogue_size"]:,} products and {data["train_pair_rules"]:,} directional pair rules. Training ends {data["train_end"]}; testing starts {data["test_start"]}. Seed {data["seed"]}; {data["bootstrap_resamples"]} invoice bootstrap resamples. Support threshold {data["min_support"]}; confidence threshold {data["min_confidence"]}. These are fixed settings, not tuned on test.

## Results
Hit rate, precision and coverage are percentages. Intervals are unclustered percentile bootstrap intervals for hit rate, not tests of pairwise model differences.

| Method | K | Hit rate (%) | 95% bootstrap interval (%) | Precision (%) | Catalogue coverage (%) |
|---|---:|---:|---|---:|---:|
{chr(10).join(rows)}

Because each invoice has exactly one hidden relevant product, Recall@K equals HitRate@K and Precision@K equals HitRate@K/K.

## Interpretation
This experiment supports a descriptive, single-window comparison on a retrospective basket-completion protocol. It does not establish online uplift, causal impact, next-purchase accuracy, or statistically significant superiority. The reported intervals do not account for repeated customers. Random concealment is performed only among products seen in training and excludes cold-start invoices. Pair-rule ranking includes a popularity fallback. See docs/HELDOUT_TOPK_PROTOCOL.md for reproducibility.

## Reproduction
Run `python -m src.heldout_topk "data/raw/Online Retail.xlsx" --bootstrap 500 --output reports/generated/heldout_topk.json` followed by `python -m src.evidence_report reports/generated/heldout_topk.json --output-dir reports/generated/publication`. Use the official UCI dataset acquisition workflow.

## Further evidence needed for journal submission
Paired model-difference intervals/tests, multiple time-origin splits, customer-clustered uncertainty, hyperparameter sensitivity, additional rule-ranking methods and cold-start analysis. Avoid describing this single-window study as fully publication validated.
"""
    (output_dir / "EVALUATION_REPORT.md").write_text(report, encoding="utf-8")
    print(f"Generated evidence-derived report and figures in {output_dir}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path)
    p.add_argument("--output-dir", type=Path, default=Path("reports/generated/publication"))
    args = p.parse_args()
    generate(args.source, args.output_dir)

if __name__ == "__main__":
    main()
