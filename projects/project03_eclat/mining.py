"""Project 03: vertical TID-set Eclat and equivalence-checked retail mining."""
from __future__ import annotations
import argparse
import json
import time
import tracemalloc
from pathlib import Path
import numpy as np
from src.data import load_transactions, clean_transactions
from src.chronological_holdout import invoice_matrix
from projects.project02_fpgrowth.benchmark import sparse_binary_frame,run_miner,canonical_itemsets

def eclat_itemsets(matrix, codes, min_support=.01, max_len=2):
    """Depth-first vertical TID-set intersections; exact fractional supports."""
    if not 0<min_support<=1 or max_len<1:
        raise ValueError("Invalid mining configuration")
    x=matrix.tocsc()
    n=x.shape[0]
    minimum=int(np.ceil(min_support*n-1e-12))
    base=[]
    for j in range(x.shape[1]):
        tids=set(map(int,x.indices[x.indptr[j]:x.indptr[j+1]]))
        if len(tids)>=minimum:
            base.append(((str(codes[j]),),tids))
    base.sort(key=lambda z:z[0])
    result={}
    def recurse(prefix, candidates):
        for i,(suffix,tids) in enumerate(candidates):
            item=prefix+suffix
            result[item]=len(tids)/n
            if len(item)>=max_len:
                continue
            children=[]
            for next_suffix,next_tids in candidates[i+1:]:
                intersection=tids&next_tids
                if len(intersection)>=minimum:
                    children.append((next_suffix,intersection))
            recurse(item,children)
    recurse((),base)
    return result

def timed_eclat(matrix,codes,support,max_len):
    tracemalloc.start()
    t0=time.perf_counter()
    result=eclat_itemsets(matrix,codes,support,max_len)
    elapsed=time.perf_counter()-t0
    _,peak=tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result,elapsed,peak/1048576

def evaluate(workbook,origins=(.5,.6,.7),supports=(.01,.02),max_len=2):
    cleaned,cleaning=clean_transactions(load_transactions(Path(workbook)))
    grouped,codes,matrix=invoice_matrix(cleaned)
    rows=[]
    for origin in origins:
        cut=int(matrix.shape[0]*origin)
        train=matrix[:cut]
        active=np.asarray(train.sum(axis=0)).ravel()>0
        active_codes=np.asarray(codes)[active]
        reduced=train[:,active]
        frame=sparse_binary_frame(reduced,active_codes)
        for support in supports:
            e,e_sec,e_peak=timed_eclat(reduced,active_codes,support,max_len)
            a,a_sec,a_peak=run_miner(frame,"apriori",support,max_len)
            f,f_sec,f_peak=run_miner(frame,"fpgrowth",support,max_len)
            for name,other in (("Apriori",canonical_itemsets(a)),("FP-Growth",canonical_itemsets(f))):
                if e.keys()!=other.keys() or any(abs(e[k]-other[k])>1e-10 for k in e):
                    raise AssertionError(f"Eclat differs from {name} at origin={origin} support={support}")
            rows.append({"origin":origin,"support":support,"train_invoices":cut,
                "train_products":len(active_codes),"itemsets":len(e),
                "exact_three_way_equivalence":True,"eclat_seconds":e_sec,
                "apriori_seconds":a_sec,"fpgrowth_seconds":f_sec,
                "eclat_tracemalloc_mib":e_peak,"apriori_tracemalloc_mib":a_peak,
                "fpgrowth_tracemalloc_mib":f_peak})
    return {"project":"03 Eclat","dataset":"Official UCI Online Retail UK invoices",
            "design":"Chronological train-only 50/60/70% prefixes; 1/2% support; max pair length",
            "limitations":["Single-run timings exploratory and order-dependent",
                "Python tracemalloc is not resident memory","Only itemsets up to size two"],
            "cleaning":cleaning,"results":rows}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("workbook",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project03_mining.json"))
    a=p.parse_args()
    result=evaluate(a.workbook)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["results"],indent=2))
if __name__=="__main__":
    main()
