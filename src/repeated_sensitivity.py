"""Repeated real-UCI Apriori/mlxtend equivalence and sensitivity experiment.

Runs the established real-data experiment; records actual observations only.
Each run independently asserts equality of itemsets and supports.
"""
from __future__ import annotations
import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from src.real_data_apriori_experiment import run

CONFIGS = [(100, 0.05), (100, 0.10), (300, 0.05),
           (300, 0.10), (1000, 0.05), (1000, 0.10)]

def execute(workbook: Path, repeats: int = 2):
    if repeats < 2:
        raise ValueError("At least two repeats are required")
    results = []
    for invoices, support in CONFIGS:
        observations = []
        for repeat in range(repeats):
            result = run(workbook, max_invoices=invoices, min_support=support, max_len=3)
            if not result["itemsets_and_support_match"]:
                raise AssertionError("Real UCI mlxtend mismatch")
            observations.append({
                "repeat": repeat + 1,
                "independent_seconds": result["independent_performance"]["seconds"],
                "mlxtend_seconds": result["mlxtend_performance"]["seconds"],
                "independent_python_peak_mib": result["independent_performance"]["python_tracemalloc_peak_mib"],
                "mlxtend_python_peak_mib": result["mlxtend_performance"]["python_tracemalloc_peak_mib"],
            })
        results.append({
            "invoices": invoices,
            "min_support": support,
            "max_len": 3,
            "frequent_itemsets": result["independent_itemsets"],
            "reference_itemsets": result["mlxtend_itemsets"],
            "itemsets_and_support_match": True,
            "observations": observations,
            "median_independent_seconds": statistics.median(o["independent_seconds"] for o in observations),
            "median_mlxtend_seconds": statistics.median(o["mlxtend_seconds"] for o in observations),
        })
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Official UCI Online Retail workbook downloaded in GitHub Actions",
        "protocol": "First N invoices in chronological 70% UK training split; test untouched",
        "repeats_per_configuration": repeats,
        "limitations": [
            "Two repeats are exploratory and not enough for statistical inference.",
            "Workbook loading, cleaning and basket construction are excluded from timed mining.",
            "tracemalloc measures Python-tracked allocations, not process RSS.",
            "Support sensitivity uses only 5% and 10%; lower thresholds need resource feasibility checks.",
            "This is itemset equivalence, not held-out recommendation accuracy."
        ],
        "configurations": results,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("workbook", type=Path)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--output", type=Path, default=Path("reports/generated/repeated_sensitivity.json"))
    args = p.parse_args()
    result = execute(args.workbook, args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
