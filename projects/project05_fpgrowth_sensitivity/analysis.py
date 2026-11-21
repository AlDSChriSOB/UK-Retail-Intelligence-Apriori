"""Project 05: FP-Growth support/confidence sensitivity on real UK invoices."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from src.data import load_transactions,clean_transactions
from src.chronological_holdout import invoice_matrix
from src.robust_benchmark import rank_items
from projects.project02_fpgrowth.benchmark import sparse_binary_frame,run_miner,canonical_itemsets
from projects.project02_fpgrowth.recommendation import pair_rules_from_itemsets

def evaluate(workbook,origins=(.5,.6,.7),horizon=.1,
             supports=(.005,.01,.02),confidences=(.1,.2,.4),seed=2026):
    cleaned,summary=clean_transactions(load_transactions(Path(workbook)))
    grouped,codes,matrix=invoice_matrix(cleaned)
    folds=[]
    for fi,origin in enumerate(origins):
        cut,end=int(matrix.shape[0]*origin),int(matrix.shape[0]*(origin+horizon))
        if not 0<cut<end<=matrix.shape[0]:
            raise ValueError("Invalid chronological fold")
        train,test=matrix[:cut],matrix[cut:end]
        freq=np.asarray(train.sum(axis=0)).ravel()
        active=freq>0
        frame=sparse_binary_frame(train[:,active],np.asarray(codes)[active])
        co=(train.astype(np.int64).T@train.astype(np.int64)).tocsr()
        co.setdiag(0);co.eliminate_zeros()
        rng=np.random.default_rng(seed+fi)
        trials=[]
        for i in range(test.shape[0]):
            basket=set(test.indices[test.indptr[i]:test.indptr[i+1]])
            eligible=sorted(j for j in basket if active[j])
            if len(eligible)<2:
                continue
            target=int(rng.choice(eligible))
            trials.append((basket-{target},target))
        if not trials:
            raise ValueError("No eligible test invoices")
        rows=[]
        for support in supports:
            itemsets,elapsed,peak=run_miner(frame,"fpgrowth",support,2)
            canonical=canonical_itemsets(itemsets)
            for confidence in confidences:
                rules=pair_rules_from_itemsets(canonical,codes,cut,confidence)
                hits5=hits10=0
                for context,target in trials:
                    ranking=rank_items(context,freq,co,rules,"apriori",10)
                    hits5+=int(target in ranking[:5])
                    hits10+=int(target in ranking[:10])
                rows.append({"support":support,"confidence":confidence,
                             "itemsets":len(canonical),"directional_rules":int(rules.nnz),
                             "hit_rate_5":hits5/len(trials),"hit_rate_10":hits10/len(trials),
                             "mining_seconds":elapsed,"python_alloc_peak_mib":peak})
        folds.append({"origin":origin,"train_invoices":cut,"test_invoices":end-cut,
                      "eligible":len(trials),"train_end":str(grouped.iloc[cut-1]["date"]),
                      "test_start":str(grouped.iloc[cut]["date"]),"grid":rows})
    return {"project":"05 FP-Growth threshold sensitivity","dataset":"Official UCI Online Retail, positive UK invoices",
            "protocol":"Three chronological training prefixes with next 10% held-out invoice windows; 3x3 fixed support/confidence grid; deterministic masked training-known item",
            "selection_policy":"All nine configurations are reported; no test-selected winner claimed",
            "cleaning":summary,"seed":seed,"folds":folds,
            "limitations":["Within-invoice masked-item recovery, not future purchasing",
                           "Test grid is descriptive sensitivity analysis, not unbiased tuned-model evaluation",
                           "Repeated settings and folds are statistically dependent",
                           "Single-run timings and Python allocation are exploratory"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("workbook",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project05_sensitivity.json"))
    a=p.parse_args()
    result=evaluate(a.workbook)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps([{"origin":f["origin"],"eligible":f["eligible"],"grid":f["grid"]} for f in result["folds"]],indent=2))
if __name__=="__main__":
    main()
