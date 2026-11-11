"""Chronological holdout validation of pairwise association rules.

Uses the original dated UCI Online Retail workbook. Rules are mined on the
earliest 70% of UK invoices only; support/confidence/lift are measured again
on later invoices, without using them for rule selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.data import clean_transactions, load_transactions


def invoice_matrix(df):
    grouped = df.groupby("InvoiceNo", sort=False).agg(
        date=("InvoiceDate", "min"),
    ).reset_index().sort_values(["date", "InvoiceNo"], kind="stable")
    codes = sorted(df["StockCode"].unique())
    inv_ids = grouped["InvoiceNo"].tolist()
    invoice_to_row = {inv: i for i, inv in enumerate(inv_ids)}
    code_to_col = {code: j for j, code in enumerate(codes)}
    pairs = df[["InvoiceNo", "StockCode"]].drop_duplicates()
    rows = pairs["InvoiceNo"].map(invoice_to_row).to_numpy()
    cols = pairs["StockCode"].map(code_to_col).to_numpy()
    mat = csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(len(inv_ids), len(codes)),
    )
    return grouped, codes, mat


def rule_statistics(matrix, codes, min_support=0.01, min_confidence=0.2):
    n = matrix.shape[0]
    if n == 0:
        raise ValueError("Empty invoice population")
    item_counts = np.asarray(matrix.sum(axis=0)).ravel()
    frequent = np.flatnonzero(item_counts >= np.ceil(min_support * n))
    pair_counts = (matrix[:, frequent].T @ matrix[:, frequent]).tocoo()
    rules = []
    for i, j, both in zip(pair_counts.row, pair_counts.col, pair_counts.data):
        if i == j or both / n < min_support:
            continue
        a, b = int(frequent[i]), int(frequent[j])
        confidence = int(both) / int(item_counts[a])
        if confidence >= min_confidence:
            rules.append((a, b, int(both)))
    return rules, item_counts


def evaluate(workbook, train_fraction=0.7, min_support=0.01, min_confidence=0.2):
    original = load_transactions(Path(workbook))
    cleaned, cleaning = clean_transactions(original)
    grouped, codes, matrix = invoice_matrix(cleaned)
    cut = int(len(grouped) * train_fraction)
    if cut < 1 or cut >= len(grouped):
        raise ValueError("Train fraction yields empty train/test set")
    train, test = matrix[:cut], matrix[cut:]
    rules, train_item_counts = rule_statistics(train, codes, min_support, min_confidence)
    test_counts = np.asarray(test.sum(axis=0)).ravel()
    test_cooccurrence = (test.T @ test).tocsr()
    evaluated = []
    for a, b, both_train in rules:
        both_test = int(test_cooccurrence[a, b])
        a_test = int(test_counts[a])
        b_test = int(test_counts[b])
        train_conf = both_train / int(train_item_counts[a])
        test_conf = both_test / a_test if a_test else None
        test_lift = (test_conf / (b_test / test.shape[0])
                     if test_conf is not None and b_test else None)
        evaluated.append({
            "antecedent": codes[a], "consequent": codes[b],
            "train_joint_invoices": both_train,
            "train_confidence": train_conf,
            "test_joint_invoices": both_test,
            "test_antecedent_invoices": a_test,
            "test_consequent_invoices": b_test,
            "test_confidence": test_conf,
            "test_lift": test_lift,
        })
    observed = [r for r in evaluated if r["test_confidence"] is not None]
    return {
        "dataset": "UCI Online Retail; United Kingdom positive non-cancelled sales",
        "cleaning": cleaning,
        "train_fraction": train_fraction,
        "min_support": min_support,
        "min_confidence": min_confidence,
        "split": {
            "train_invoices": cut, "test_invoices": test.shape[0],
            "last_train_invoice_date": str(grouped.iloc[cut - 1]["date"]),
            "first_test_invoice_date": str(grouped.iloc[cut]["date"]),
        },
        "rule_summary": {
            "train_qualifying_directional_pair_rules": len(rules),
            "rules_with_test_antecedent": len(observed),
            "rules_with_nonzero_test_cooccurrence": sum(r["test_joint_invoices"] > 0 for r in evaluated),
            "mean_train_confidence": float(np.mean([r["train_confidence"] for r in evaluated])) if evaluated else None,
            "mean_test_confidence_on_observed_antecedents": float(np.mean([r["test_confidence"] for r in observed])) if observed else None,
            "median_test_lift_on_observed_antecedents": float(np.median([r["test_lift"] for r in observed if r["test_lift"] is not None])) if any(r["test_lift"] is not None for r in observed) else None,
        },
        "top_20_train_rules_by_confidence": sorted(evaluated, key=lambda r: -r["train_confidence"])[:20],
        "notes": [
            "Invoice-level chronological holdout, not a customer-level split.",
            "Test metrics are descriptive rule replication, not recommendation precision@K.",
            "Identical timestamps may straddle the 70% boundary; inspect dates.",
            "No hyperparameter tuning on the test period.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook", help="Path to original Online Retail.xlsx")
    parser.add_argument("--output", default="reports/generated/chronological_pair_holdout.json")
    args = parser.parse_args()
    report = evaluate(args.workbook)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"split": report["split"], "rule_summary": report["rule_summary"]}, indent=2))


if __name__ == "__main__":
    main()
