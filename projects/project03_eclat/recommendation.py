"""Project 03: Eclat-derived rules, matched basket completion and paired inference."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from src.data import clean_transactions,load_transactions
from src.chronological_holdout import invoice_matrix
from src.temporal_clustered import customer_clusters,paired_cluster_interval
from src.robust_benchmark import rank_items
from projects.project02_fpgrowth.benchmark import sparse_binary_frame,run_miner,canonical_itemsets
from projects.project02_fpgrowth.recommendation import pair_rules_from_itemsets
from projects.project03_eclat.mining import eclat_itemsets

METHODS=("eclat","apriori","fpgrowth","cosine","popularity")

def evaluate(workbook,origins=(.5,.6,.7),horizon=.1,support=.01,
             confidence=.2,k_values=(5,10),bootstrap=300,seed=2026):
    if bootstrap<1:
        raise ValueError("Bootstrap must be positive")
    cleaned,cleaning=clean_transactions(load_transactions(Path(workbook)))
    grouped,codes,matrix=invoice_matrix(cleaned)
    clusters=customer_clusters(cleaned,grouped)
    folds=[]
    for fi,origin in enumerate(origins):
        cut,end=int(matrix.shape[0]*origin),int(matrix.shape[0]*(origin+horizon))
        if not 0<cut<end<=matrix.shape[0]:
            raise ValueError("Invalid fold")
        train,test=matrix[:cut],matrix[cut:end]
        freq=np.asarray(train.sum(axis=0)).ravel()
        active=freq>0
        reduced=train[:,active]
        frame=sparse_binary_frame(reduced,np.asarray(codes)[active])
        e=eclat_itemsets(reduced,np.asarray(codes)[active],support)
        a=canonical_itemsets(run_miner(frame,"apriori",support)[0])
        f=canonical_itemsets(run_miner(frame,"fpgrowth",support)[0])
        if e.keys()!=a.keys() or e.keys()!=f.keys() or any(abs(e[k]-a[k])>1e-10 or abs(e[k]-f[k])>1e-10 for k in e):
            raise AssertionError("Three miners disagree")
        rules={m:pair_rules_from_itemsets(itemsets,codes,cut,confidence)
               for m,itemsets in (("eclat",e),("apriori",a),("fpgrowth",f))}
        if (rules["eclat"]!=rules["apriori"]).nnz or (rules["eclat"]!=rules["fpgrowth"]).nnz:
            raise AssertionError("Three rule matrices differ")
        co=(train.astype(np.int64).T@train.astype(np.int64)).tocsr()
        co.setdiag(0);co.eliminate_zeros()
        rng=np.random.default_rng(seed+fi)
        hits={m:{k:[] for k in k_values} for m in METHODS}
        ids=[]
        for i in range(test.shape[0]):
            basket=set(test.indices[test.indptr[i]:test.indptr[i+1]])
            eligible=sorted(j for j in basket if active[j])
            if len(eligible)<2:
                continue
            target=int(rng.choice(eligible))
            context=basket-{target}
            ids.append(clusters[cut+i])
            ranked={}
            for method in METHODS:
                method_for_rank="apriori" if method in rules else method
                ranked[method]=rank_items(context,freq,co,rules["eclat"] if method not in rules else rules[method],
                                          method_for_rank,max(k_values))
            if not (ranked["eclat"]==ranked["apriori"]==ranked["fpgrowth"]):
                raise AssertionError("Three mining-based rankings differ")
            for method in METHODS:
                for k in k_values:
                    hits[method][k].append(int(target in ranked[method][:k]))
        if not ids:
            raise ValueError("No eligible test invoices")
        metrics={}
        for k in k_values:
            row={m:float(np.mean(hits[m][k])) for m in METHODS}
            for comp in ("apriori","fpgrowth","cosine","popularity"):
                delta=np.asarray(hits["eclat"][k],float)-np.asarray(hits[comp][k],float)
                row["eclat_minus_"+comp]={
                    "difference":float(delta.mean()),
                    "cluster_bootstrap_95pct_ci":paired_cluster_interval(
                        delta,ids,bootstrap,seed+fi*1000+k*10+METHODS.index(comp))}
            metrics[str(k)]=row
        folds.append({"origin":origin,"train_invoices":cut,"test_invoices":end-cut,
                      "eligible":len(ids),"clusters":len(set(ids)),
                      "train_end":str(grouped.iloc[cut-1]["date"]),
                      "test_start":str(grouped.iloc[cut]["date"]),
                      "itemsets":len(e),"directional_rules":int(rules["eclat"].nnz),
                      "three_way_itemset_and_prediction_equivalence":True,
                      "metrics":metrics})
    return {"project":"03 Eclat","source":"Official UCI Online Retail cleaned UK transactions",
            "protocol":"Chronological expanding prefixes, next non-overlapping 10% test windows; identical concealed training-known product",
            "support":support,"confidence":confidence,"bootstrap":bootstrap,
            "seed":seed,"cleaning":cleaning,"folds":folds,
            "limitations":["Within-invoice masked-item recovery, not next-order prediction",
                "Unadjusted exploratory customer-cluster bootstrap intervals",
                "Missing CustomerID handled as separate invoice clusters",
                "Three neighboring windows not independent external replications",
                "Equivalent itemsets with same scoring must produce identical recommendations"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("workbook",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project03_recommendation.json"))
    a=p.parse_args()
    result=evaluate(a.workbook)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps([{"origin":f["origin"],"eligible":f["eligible"],"metrics":f["metrics"]} for f in result["folds"]],indent=2))
if __name__=="__main__":
    main()
