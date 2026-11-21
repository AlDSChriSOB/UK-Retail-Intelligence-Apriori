"""Project 02: leakage-controlled FP-Growth versus Apriori itemset equivalence and cost.

Run: python -m projects.project02_fpgrowth.benchmark "data/raw/Online Retail.xlsx"
"""
from __future__ import annotations
import argparse
import json
import time
import tracemalloc
from pathlib import Path
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, fpgrowth
from src.data import load_transactions, clean_transactions
from src.chronological_holdout import invoice_matrix

def sparse_binary_frame(matrix, codes):
    """Train-only boolean sparse invoice-item table (no dense full catalogue matrix)."""
    return pd.DataFrame.sparse.from_spmatrix(matrix.astype(bool), columns=[str(x) for x in codes])

def canonical_itemsets(result):
    return {tuple(sorted(str(x) for x in row.itemsets)): float(row.support)
            for row in result.itertuples(index=False)}

def run_miner(table, algorithm, min_support, max_len=2):
    func = {"apriori": apriori, "fpgrowth": fpgrowth}[algorithm]
    tracemalloc.start()
    start = time.perf_counter()
    result = func(table, min_support=min_support, use_colnames=True,
                  max_len=max_len, low_memory=True) if algorithm == "apriori" else func(
                      table, min_support=min_support, use_colnames=True, max_len=max_len)
    seconds = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, seconds, peak / (1024 * 1024)

def evaluate(workbook, origins=(.5,.6,.7), supports=(.01,.02), max_len=2):
    cleaned, summary = clean_transactions(load_transactions(Path(workbook)))
    grouped, codes, matrix = invoice_matrix(cleaned)
    rows = []
    for origin in origins:
        n_train = int(matrix.shape[0] * origin)
        train = matrix[:n_train]
        active = np.asarray(train.sum(axis=0)).ravel() > 0
        train_frame = sparse_binary_frame(train[:,active], np.asarray(codes)[active])
        for support in supports:
            a, a_sec, a_mb = run_miner(train_frame, "apriori", support, max_len)
            f, f_sec, f_mb = run_miner(train_frame, "fpgrowth", support, max_len)
            aa, ff = canonical_itemsets(a), canonical_itemsets(f)
            missing_a, missing_f = set(aa)-set(ff), set(ff)-set(aa)
            largest_delta = max((abs(aa[k]-ff[k]) for k in set(aa)&set(ff)), default=0.0)
            if missing_a or missing_f or largest_delta > 1e-10:
                raise AssertionError(f"Frequent-itemset mismatch: Apriori-only={len(missing_a)}, FP-only={len(missing_f)}, delta={largest_delta}")
            rows.append({"train_origin":origin,"train_invoices":n_train,
                "train_products":int(active.sum()),"min_support":support,"max_len":max_len,
                "apriori_itemsets":len(aa),"fpgrowth_itemsets":len(ff),
                "equal_itemsets_and_support":True,"max_support_abs_delta":largest_delta,
                "apriori_seconds":a_sec,"fpgrowth_seconds":f_sec,
                "apriori_tracemalloc_peak_mib":a_mb,"fpgrowth_tracemalloc_peak_mib":f_mb,
                "fpgrowth_speedup_ratio":a_sec/f_sec if f_sec else None})
    return {"project":"02 — FP-Growth versus Apriori","source":"Official UCI Online Retail",
            "method":"Chronological train-only mining; max_len=2; pandas sparse Boolean incidence; exact itemset-support equivalence",
            "memory_note":"tracemalloc measures Python-tracked allocations, not process peak RSS",
            "timing_note":"single-process sequential timings are exploratory, not hardware-independent speed claims",
            "cleaning":summary,"results":rows}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("workbook",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project02_fpgrowth.json"))
    args=p.parse_args()
    result=evaluate(args.workbook)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["results"],indent=2))
if __name__=="__main__":
    main()
