"""Real-data chronological holdout analysis of mined pairwise rules.

Reuses the established train-only rule selection in chronological_holdout.
This is descriptive temporal replication, not precision@K or causal inference.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from src.chronological_holdout import evaluate

def analyze(workbook: Path):
    base = evaluate(workbook)
    rules = []
    for r in base["top_20_train_rules_by_confidence"]:
        train = r["train_confidence"]
        test = r["test_confidence"]
        rules.append({
            "antecedent": r["antecedent"],
            "consequent": r["consequent"],
            "train_confidence": train,
            "test_confidence": test,
            "confidence_change": (test - train) if test is not None else None,
            "test_lift": r["test_lift"],
            "test_joint_invoices": r["test_joint_invoices"],
            "test_antecedent_invoices": r["test_antecedent_invoices"],
            "test_antecedent_observed": test is not None,
        })
    return {
        "source": "UCI Online Retail UK cleaned transactions",
        "split": base["split"],
        "train_rule_selection": {
            "min_support": base["min_support"],
            "min_confidence": base["min_confidence"],
            "train_qualifying_directional_pair_rules": base["rule_summary"]["train_qualifying_directional_pair_rules"],
        },
        "all_train_rules_test_summary": base["rule_summary"],
        "top_20_train_rules_holdout": rules,
        "limitations": [
            "Rule ranking and selection use training invoices only.",
            "Test confidence undefined when antecedent never occurs in test.",
            "Top 20 table is not representative of all rules.",
            "This is descriptive temporal replication, not precision@K, CTR, or causal impact.",
            "Invoice-level split may place identical timestamps across boundary.",
        ],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/generated/rule_stability.json"))
    args = parser.parse_args()
    result = analyze(args.workbook)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"split": result["split"], "summary": result["all_train_rules_test_summary"],
                      "top_20_rules": len(result["top_20_train_rules_holdout"])}, indent=2))

if __name__ == "__main__":
    main()
