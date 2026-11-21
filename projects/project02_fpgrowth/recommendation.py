"""Project 02: matched FP-Growth/Apriori recommendation and paired uncertainty.

Both miners must produce identical training itemsets and identical predictions.
The evaluation uses genuine UCI records and never tunes on held-out invoices.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from src.data import clean_transactions, load_transactions
from src.chronological_holdout import invoice_matrix
from src.temporal_clustered import customer_clusters, paired_cluster_interval
from src.robust_benchmark import rank_items
from projects.project02_fpgrowth.benchmark import sparse_binary_frame, run_miner, canonical_itemsets

METHODS = ("fpgrowth", "apriori", "cosine", "popularity")

def pair_rules_from_itemsets(itemsets, codes, n_invoices, min_confidence):
    """Create directional pair-confidence CSR scores from mined singleton/pair supports."""
    index = {str(code):i for i,code in enumerate(codes)}
    singles = {item[0]:support for item,support in itemsets.items() if len(item)==1}
    rows,cols,weights = [],[],[]
    for item,support in itemsets.items():
        if len(item)!=2:
            continue
        a,b=item
        for antecedent,consequent in ((a,b),(b,a)):
            denominator=singles[antecedent]
            confidence=support/denominator
            if confidence+1e-12>=min_confidence:
                rows.append(index[antecedent]);cols.append(index[consequent]);weights.append(confidence)
    return csr_matrix((weights,(rows,cols)),shape=(len(codes),len(codes)))

def evaluate(workbook, origins=(.5,.6,.7), horizon=.1, support=.01,
             confidence=.2, k_values=(5,10), bootstrap=300, seed=2026):
    if bootstrap < 1 or not 0<support<=1 or not 0<confidence<=1:
        raise ValueError("Invalid benchmark settings")
    cleaned,cleaning=clean_transactions(load_transactions(Path(workbook)))
    grouped,codes,matrix=invoice_matrix(cleaned)
    clusters=customer_clusters(cleaned,grouped)
    folds=[]
    for fold_index,origin in enumerate(origins):
        cut,end=int(matrix.shape[0]*origin),int(matrix.shape[0]*(origin+horizon))
        if not 0<cut<end<=matrix.shape[0]:
            raise ValueError("Invalid temporal fold")
        train,test=matrix[:cut],matrix[cut:end]
        freq=np.asarray(train.sum(axis=0)).ravel()
        active=freq>0
        frame=sparse_binary_frame(train[:,active],np.asarray(codes)[active])
        apriori_result,a_time,a_peak=run_miner(frame,"apriori",support)
        fp_result,fp_time,fp_peak=run_miner(frame,"fpgrowth",support)
        a_items,f_items=canonical_itemsets(apriori_result),canonical_itemsets(fp_result)
        if set(a_items)!=set(f_items) or any(abs(a_items[k]-f_items[k])>1e-10 for k in a_items):
            raise AssertionError("FP-Growth and Apriori disagree on frequent itemsets")
        arules=pair_rules_from_itemsets(a_items,codes,cut,confidence)
        frules=pair_rules_from_itemsets(f_items,codes,cut,confidence)
        if (arules!=frules).nnz:
            raise AssertionError("FP-Growth and Apriori directional rules differ")
        co=(train.astype(np.int64).T@train.astype(np.int64)).tocsr()
        co.setdiag(0);co.eliminate_zeros()
        rng=np.random.default_rng(seed+fold_index)
        outcomes={m:{k:[] for k in k_values} for m in METHODS}
        ids=[]
        for i in range(test.shape[0]):
            observed=set(test.indices[test.indptr[i]:test.indptr[i+1]])
            known=sorted(j for j in observed if active[j])
            if len(known)<2:
                continue
            target=int(rng.choice(known))
            context=observed-{target}
            ids.append(clusters[cut+i])
            rankings={
                "fpgrowth":rank_items(context,freq,co,frules,"apriori",max(k_values)),
                "apriori":rank_items(context,freq,co,arules,"apriori",max(k_values)),
                "cosine":rank_items(context,freq,co,arules,"cosine",max(k_values)),
                "popularity":rank_items(context,freq,co,arules,"popularity",max(k_values))
            }
            if rankings["fpgrowth"]!=rankings["apriori"]:
                raise AssertionError("Matched miner predictions differ")
            for method in METHODS:
                for k in k_values:
                    outcomes[method][k].append(int(target in rankings[method][:k]))
        if not ids:
            raise ValueError("No eligible test invoices")
        metrics={}
        for k in k_values:
            row={method:float(np.mean(outcomes[method][k])) for method in METHODS}
            for competitor in ("apriori","cosine","popularity"):
                diff=np.asarray(outcomes["fpgrowth"][k],float)-np.asarray(outcomes[competitor][k],float)
                row["fpgrowth_minus_"+competitor]={
                    "difference":float(diff.mean()),
                    "customer_cluster_95pct_ci":paired_cluster_interval(
                        diff,ids,bootstrap,seed+fold_index*100+k*10+METHODS.index(competitor))}
            metrics[str(k)]=row
        folds.append({"origin":origin,"train_invoices":cut,"test_invoices":end-cut,
            "eligible":len(ids),"clusters":len(set(ids)),"train_end":str(grouped.iloc[cut-1]["date"]),
            "test_start":str(grouped.iloc[cut]["date"]),
            "itemsets":len(a_items),"directional_rules":int(arules.nnz),
            "exact_itemset_equivalence":True,"exact_prediction_equivalence":True,
            "apriori_seconds":a_time,"fpgrowth_seconds":fp_time,
            "apriori_tracemalloc_peak_mib":a_peak,"fpgrowth_tracemalloc_peak_mib":fp_peak,
            "metrics":metrics})
    return {"source":"Official UCI Online Retail; cleaned positive UK invoices",
        "protocol":"Three expanding chronological training origins, following non-overlapping 10% invoice windows; one concealed training-known item",
        "support":support,"confidence":confidence,"k_values":list(k_values),
        "bootstrap":bootstrap,"seed":seed,"cleaning":cleaning,"folds":folds,
        "limitations":["Within-invoice basket completion, not future purchase prediction",
            "Known customer IDs clustered within fold; missing IDs invoice-specific",
            "Exploratory percentile bootstrap intervals unadjusted for multiplicity",
            "Tracemalloc measures Python allocations, not process RSS",
            "FP-Growth and Apriori use identical pair-confidence scoring; accuracy equivalence is expected",
            "Single timings are hardware- and execution-order-dependent"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("workbook",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project02_recommendation.json"))
    p.add_argument("--bootstrap",type=int,default=300)
    a=p.parse_args()
    result=evaluate(a.workbook,bootstrap=a.bootstrap)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps([{"origin":f["origin"],"eligible":f["eligible"],"metrics":f["metrics"]}
                      for f in result["folds"]],indent=2))
if __name__=="__main__":
    main()
