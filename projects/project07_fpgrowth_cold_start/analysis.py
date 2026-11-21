"""Project 07: FP-Growth robustness by training-item frequency and cold start."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from src.data import clean_transactions,load_transactions
from src.chronological_holdout import invoice_matrix
from src.robust_benchmark import rank_items
from projects.project02_fpgrowth.benchmark import sparse_binary_frame,run_miner,canonical_itemsets
from projects.project02_fpgrowth.recommendation import pair_rules_from_itemsets

METHODS=("fpgrowth","cosine","popularity")
STRATA=("unseen","rare","mid","popular")

def frequency_strata(freq):
    """Train-only item frequency quantiles; zero-frequency items remain unseen."""
    freq=np.asarray(freq)
    if freq.ndim!=1 or np.any(freq<0):raise ValueError("Expected nonnegative vector")
    positive=freq[freq>0]
    if not len(positive):raise ValueError("No training products")
    q1,q2=np.quantile(positive,[1/3,2/3])
    labels=np.full(len(freq),"unseen",dtype=object)
    labels[(freq>0)&(freq<=q1)]="rare"
    labels[(freq>q1)&(freq<=q2)]="mid"
    labels[freq>q2]="popular"
    return labels,{"tertile_1":float(q1),"tertile_2":float(q2)}

def evaluate(workbook,origins=(.5,.6,.7),horizon=.1,support=.01,confidence=.2,seed=2026):
    if not 0<support<=1 or not 0<confidence<=1:raise ValueError("Invalid thresholds")
    cleaned,summary=clean_transactions(load_transactions(Path(workbook)))
    grouped,codes,matrix=invoice_matrix(cleaned)
    folds=[]
    for fi,origin in enumerate(origins):
        cut,end=int(matrix.shape[0]*origin),int(matrix.shape[0]*(origin+horizon))
        if not 0<cut<end<=matrix.shape[0]:raise ValueError("Invalid fold")
        train,test=matrix[:cut],matrix[cut:end]
        freq=np.asarray(train.sum(axis=0)).ravel()
        labels,quantiles=frequency_strata(freq)
        active=freq>0
        frame=sparse_binary_frame(train[:,active],np.asarray(codes)[active])
        itemsets,elapsed,peak=run_miner(frame,"fpgrowth",support,2)
        rules=pair_rules_from_itemsets(canonical_itemsets(itemsets),codes,cut,confidence)
        co=(train.astype(np.int64).T@train.astype(np.int64)).tocsr()
        co.setdiag(0);co.eliminate_zeros()
        rng=np.random.default_rng(seed+fi)
        counts={s:0 for s in STRATA}
        hits={s:{m:{5:0,10:0} for m in METHODS} for s in STRATA}
        for i in range(test.shape[0]):
            basket=set(test.indices[test.indptr[i]:test.indptr[i+1]])
            if len(basket)<2:continue
            target=int(rng.choice(sorted(basket)))
            context=basket-{target}
            s=labels[target];counts[s]+=1
            for m in METHODS:
                ranked=rank_items(context,freq,co,rules,"apriori" if m=="fpgrowth" else m,10)
                for k in (5,10):
                    hits[s][m][k]+=int(target in ranked[:k])
        result={}
        for s in STRATA:
            n=counts[s]
            result[s]={"eligible":n,"metrics":{m:{"hit_rate_5":hits[s][m][5]/n if n else None,
                                               "hit_rate_10":hits[s][m][10]/n if n else None}
                                           for m in METHODS}}
        folds.append({"origin":origin,"train_invoices":cut,"test_invoices":end-cut,
                      "train_end":str(grouped.iloc[cut-1]["date"]),
                      "test_start":str(grouped.iloc[cut]["date"]),
                      "quantiles":quantiles,"itemsets":len(itemsets),"rules":int(rules.nnz),
                      "mining_seconds":elapsed,"python_alloc_peak_mib":peak,
                      "strata":result})
    return {"project":"07 FP-Growth cold-start and popularity stratification",
            "source":"Official UCI Online Retail, positive UK invoices",
            "protocol":"Chronological 50/60/70% train prefixes and subsequent 10% heldout; random target from every eligible multi-item basket; strata based solely on train-item frequency",
            "methods":list(METHODS),"strata":list(STRATA),"seed":seed,"support":support,
            "confidence":confidence,"cleaning":summary,"folds":folds,
            "limitations":["Within-invoice masked-item task, not future purchasing",
                           "Random target choice is not balanced by popularity; report per-stratum denominators",
                           "Training-unseen targets cannot be ranked by train-catalogue-only methods",
                           "Tertile cutoffs vary by training fold and are not global frequency thresholds",
                           "No statistical significance or causal lift claims",
                           "Single historical retailer and exploratory execution timing"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("workbook",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project07_cold_start.json"))
    a=p.parse_args()
    result=evaluate(a.workbook)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result["folds"],indent=2))
if __name__=="__main__":main()
