"""Reproduce catalogue-description collision sensitivity using the dated workbook.

Identical primary descriptions are only *candidate* duplicate products.
Both catalogues are retrained independently on the same chronological split.
Usage: python -m src.catalogue_sensitivity "data/raw/Online Retail.xlsx"
"""
import argparse
import json
from pathlib import Path
import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np
from src.data import load_transactions, clean_transactions
from src.chronological_holdout import invoice_matrix
from src.recommendation_benchmark import benchmark


def candidate_mapping(df):
    descriptions = df.dropna(subset=["Description"])[["StockCode", "Description"]].copy()
    descriptions["normalized"] = (
        descriptions["Description"].astype(str).str.upper()
        .str.replace(r"\s+", " ", regex=True).str.strip()
    )
    descriptions = descriptions[descriptions["normalized"].ne("")]
    counts = descriptions.groupby(["StockCode", "normalized"]).size().reset_index(name="lines")
    primary = (counts.sort_values(
        ["StockCode", "lines", "normalized"], ascending=[True, False, True]
    ).drop_duplicates("StockCode"))
    groups = primary.groupby("normalized")["StockCode"].agg(list)
    duplicate_groups = {desc: codes for desc, codes in groups.items() if len(codes) > 1}
    mapping = {code: min(codes) for codes in duplicate_groups.values() for code in codes}
    return mapping, duplicate_groups


def run(workbook, seeds=(42, 7), train_fraction=0.7):
    df, cleaning = clean_transactions(load_transactions(Path(workbook)))
    mapping, groups = candidate_mapping(df)
    chronological, original_codes, original = invoice_matrix(df)
    collapsed_df = df.copy()
    collapsed_df["StockCode"] = collapsed_df["StockCode"].map(lambda c: mapping.get(c, c))
    collapsed_chronological, collapsed_codes, collapsed = invoice_matrix(collapsed_df)
    assert chronological["InvoiceNo"].tolist() == collapsed_chronological["InvoiceNo"].tolist()
    output = {
        "candidate_duplicate_description_groups": len(groups),
        "codes_in_candidate_groups": len(mapping),
        "original_codes": len(original_codes),
        "collapsed_codes": len(collapsed_codes),
        "cleaning": cleaning,
        "seeds": {},
    }
    for seed in seeds:
        output["seeds"][str(seed)] = {
            "original": benchmark(original, seed=seed, train_fraction=train_fraction),
            "collapsed": benchmark(collapsed, seed=seed, train_fraction=train_fraction),
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook")
    parser.add_argument("--output", default="reports/generated/catalogue_sensitivity.json")
    args = parser.parse_args()
    result = run(args.workbook)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({
        "candidate_duplicate_description_groups": result["candidate_duplicate_description_groups"],
        "original_codes": result["original_codes"],
        "collapsed_codes": result["collapsed_codes"],
        "seeds": {seed: {
            population: {method: result["seeds"][seed][population]["metrics"][method]["10"]["hit_rate_at_k"]
                         for method in ("popularity", "cooccurrence", "apriori")}
            for population in ("original", "collapsed")
        } for seed in result["seeds"]}
    }, indent=2))


if __name__ == "__main__":
    main()
