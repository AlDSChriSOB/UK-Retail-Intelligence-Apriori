"""Project 06: recency-window FP-Growth versus expanding-history FP-Growth."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from src.data import clean_transactions,load_transactions
from src.chronological_holdout import invoice_matrix
from src.robust_benchmark import rank_items
from src.temporal_clustered import customer_clusters,paired_cluster_interval
from projects.project02_fpgrowth.benchmark import sparse_binary_frame,run_miner,canonical_itemsets
from projects.project02_fpgrowth.recommendation import pair_rules_from_itemsets

WINDOWS=("expanding","recent_40pct","recent_20pct")

def train_slice(cut,window):
    if not cut>0:
        raise ValueError("Training prefix must be positive")
    if window=="expanding":return slice(0,cut)
    fractions={"recent_40pct":.4,"recent_20pct":.2}
    if window not in fractions:raise ValueError("Unknown history window")
    return slice(max(0,cut-int(cut*fractions[window])),cut)

def evaluate(workbook,origins=(.5,.6,.7),horizon=.1,support=.01,confidence=.2,
             bootstrap=300,seed=2026):
    if not 0<support<=1 or not 0<confidence<=1 or bootstrap<1:
        raise ValueError("Invalid hyperparameters")
    cleaned,summary=clean_transactions(load_transactions(Path(workbook)))
    grouped,codes,matrix=invoice_matrix(cleaned)
    clusters=customer_clusters(cleaned,grouped)
    folds=[]
    for fi,origin in enumerate(origins):
        cut,end=int(matrix.shape[0]*origin),int(matrix.shape[0]*(origin+horizon))
        if not 0<cut<end<=matrix.shape[0]:raise ValueError("Invalid temporal split")
        test=matrix[cut:end]
        rng=np.random.default_rng(seed+fi)
        # Common eligible baskets and hidden targets across all training histories.
        # Restrict target to products observed in the expanding history; window
        # policies may have never seen it and must handle cold items honestly.
        expanding_seen=np.asarray(matrix[:cut].sum(axis=0)).ravel()>0
        trials=[];ids=[]
        for i in range(test.shape[0]):
            observed=set(test.indices[test.indptr[i]:test.indptr[i+1]])
            eligible=sorted(j for j in observed if expanding_seen[j])
            if len(eligible)<2:continue
            target=int(rng.choice(eligible))
            trials.append((observed-{target},target))
            ids.append(clusters[cut+i])
        if not trials:raise ValueError("No eligible invoices")
        outcomes={};meta={}
        for window in WINDOWS:
            train=matrix[train_slice(cut,window)]
            freq=np.asarray(train.sum(axis=0)).ravel()
            active=freq>0
            frame=sparse_binary_frame(train[:,active],np.asarray(codes)[active])
            itemsets,elapsed,peak=run_miner(frame,"fpgrowth",support,2)
            rules=pair_rules_from_itemsets(canonical_itemsets(itemsets),codes,train.shape[0],confidence)
            co=(train.astype(np.int64).T@train.astype(np.int64)).tocsr()
            co.setdiag(0);co.eliminate_zeros()
            hits5=[];hits10=[];cold=[]
            for context,target in trials:
                ranking=rank_items(context,freq,co,rules,"apriori",10)
                hits5.append(int(target in ranking[:5]))
                hits10.append(int(target in ranking[:10]))
                cold.append(int(freq[target]==0))
            outcomes[window]={5:np.asarray(hits5,float),10:np.asarray(hits10,float)}
            meta[window]={"training_invoices":train.shape[0],
                          "training_start":str(grouped.iloc[train_slice(cut,window).start or 0]["date"]),
                          "frequent_itemsets":len(itemsets),"directional_rules":int(rules.nnz),
                          "cold_target_rate":float(np.mean(cold)),
                          "hit_rate_5":float(np.mean(hits5)),"hit_rate_10":float(np.mean(hits10)),
                          "mining_seconds":elapsed,"python_alloc_peak_mib":peak}
        comparisons={}
        for window in WINDOWS[1:]:
            comparisons[window+"_minus_expanding"]={}
            for k in (5,10):
                delta=outcomes[window][k]-outcomes["expanding"][k]
                comparisons[window+"_minus_expanding"][f"hit_rate_{k}"]={
                    "difference":float(delta.mean()),
                    "customer_cluster_95pct_ci":paired_cluster_interval(delta,ids,bootstrap,seed+fi*100+k*10+WINDOWS.index(window))}
        folds.append({"origin":origin,"train_end":str(grouped.iloc[cut-1]["date"]),
                      "test_start":str(grouped.iloc[cut]["date"]),
                      "test_invoices":end-cut,"eligible":len(trials),"clusters":len(set(ids)),
                      "windows":meta,"paired_comparisons":comparisons})
    return {"project":"06 FP-Growth recency-window robustness",
            "source":"Official UCI Online Retail; positive UK invoices",
            "protocol":"Expanding, recent 40% and recent 20% of each chronological training prefix; identical next-10%-window masked-item trials and fixed mining thresholds",
            "support":support,"confidence":confidence,"bootstrap":bootstrap,"seed":seed,
            "cleaning":summary,"folds":folds,
            "limitations":["Within-invoice completion, not future-item prediction",
                           "Training windows are invoice-count windows, not equal-duration calendar windows",
                           "Customer-cluster bootstrap is descriptive and unadjusted for multiplicity",
                           "Three adjacent holdouts are not independent replications",
                           "Window policies are prespecified; do not select a winner on test performance",
                           "Python allocation and single-run timings are exploratory"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("workbook",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project06_recency.json"))
    a=p.parse_args()
    result=evaluate(a.workbook)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["folds"],indent=2))
if __name__=="__main__":
    main()
