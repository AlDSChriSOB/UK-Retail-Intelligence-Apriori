"""Project 09: vertical tidset Eclat against Apriori and FP-Growth on genuine UCI UK invoices."""
from __future__ import annotations
import argparse,json,time,tracemalloc
from pathlib import Path
from math import ceil
import numpy as np
from src.data import clean_transactions,load_transactions
from src.chronological_holdout import invoice_matrix
from projects.project02_fpgrowth.benchmark import sparse_binary_frame,run_miner,canonical_itemsets

def eclat(matrix,codes,min_support=.01,max_len=3):
    """Depth-first vertical transaction-id intersection, exact minimum count ceil(support*N)."""
    if not 0<min_support<=1 or max_len<1:raise ValueError("Invalid mining parameters")
    matrix=matrix.tocsc()
    n=matrix.shape[0]
    if not n:raise ValueError("Empty invoice population")
    threshold=ceil(min_support*n)
    items=[]
    for j,code in enumerate(codes):
        tids=set(map(int,matrix.indices[matrix.indptr[j]:matrix.indptr[j+1]]))
        if len(tids)>=threshold:items.append((str(code),tids))
    items.sort(key=lambda x:x[0])
    result={}
    def visit(prefix,extensions):
        for i,(code,tids) in enumerate(extensions):
            itemset=prefix+(code,)
            result[itemset]=len(tids)/n
            if len(itemset)<max_len:
                suffix=[]
                for other,other_tids in extensions[i+1:]:
                    intersect=tids & other_tids
                    if len(intersect)>=threshold:suffix.append((other,intersect))
                visit(itemset,suffix)
    visit((),items)
    return result

def timed_eclat(matrix,codes,support,max_len):
    tracemalloc.start();start=time.perf_counter()
    result=eclat(matrix,codes,support,max_len)
    seconds=time.perf_counter()-start
    _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    return result,seconds,peak/1024**2

def evaluate(workbook,origins=(.5,.6,.7),supports=(.01,.02),max_len=2):
    cleaned,cleaning=clean_transactions(load_transactions(Path(workbook)))
    grouped,codes,matrix=invoice_matrix(cleaned)
    rows=[]
    for origin in origins:
        cut=int(len(grouped)*origin)
        if not 0<cut<len(grouped):raise ValueError("Invalid origin")
        train=matrix[:cut]
        active=np.asarray(train.sum(axis=0)).ravel()>0
        frame=sparse_binary_frame(train[:,active],np.asarray(codes)[active])
        for support in supports:
            e,et,em=timed_eclat(train,codes,support,max_len)
            a,at,am=run_miner(frame,"apriori",support,max_len)
            f,ft,fm=run_miner(frame,"fpgrowth",support,max_len)
            aa,ff=canonical_itemsets(a),canonical_itemsets(f)
            if set(e)!=set(aa) or set(e)!=set(ff):
                raise AssertionError(f"Itemset mismatch: Eclat={len(e)}, Apriori={len(aa)}, FP={len(ff)}")
            max_delta=max((max(abs(e[k]-aa[k]),abs(e[k]-ff[k])) for k in e),default=0.)
            if max_delta>1e-10:raise AssertionError(f"Support disagreement: {max_delta}")
            rows.append({"origin":origin,"train_invoices":cut,"train_end":str(grouped.iloc[cut-1]["date"]),
                "min_support":support,"max_len":max_len,"itemsets":len(e),
                "exact_itemset_equivalence":True,"max_support_abs_delta":max_delta,
                "eclat_seconds":et,"apriori_seconds":at,"fpgrowth_seconds":ft,
                "eclat_python_peak_mib":em,"apriori_python_peak_mib":am,"fpgrowth_python_peak_mib":fm})
    return {"project":"09 Eclat vs Apriori vs FP-Growth exact itemset and exploratory runtime benchmark",
        "source":"Official UCI Online Retail, UK positive non-cancelled invoices",
        "protocol":"Train-only 50/60/70% chronological prefixes, support 1% and 2%, max pair length 2; exact itemset and support equivalence",
        "cleaning":cleaning,"rows":rows,"limitations":[
            "Python-set Eclat is a transparent reference implementation, not optimized C/C++ Eclat",
            "Single sequential timing measurements are order-, runtime- and hardware-dependent",
            "tracemalloc measures Python-tracked allocations rather than peak process RSS",
            "Equivalence concerns frequent itemsets, not prospective recommendation accuracy",
            "No inferential speedup or production-scale efficiency claim"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("workbook",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project09_eclat.json"))
    a=p.parse_args();report=evaluate(a.workbook)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report["rows"],indent=2))
if __name__=="__main__":main()
