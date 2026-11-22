"""Project 08: leakage-controlled customer RFM segmentation and FP-Growth."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.data import load_transactions,clean_transactions
from src.chronological_holdout import invoice_matrix
from src.robust_benchmark import rank_items
from projects.project02_fpgrowth.benchmark import sparse_binary_frame,run_miner,canonical_itemsets
from projects.project02_fpgrowth.recommendation import pair_rules_from_itemsets

POLICIES=("global_fpgrowth","segment_fpgrowth","global_popularity")
SEGMENTS=("low","middle","high")

def rfm_training_only(cleaned,grouped,cut):
    """Derive RFM from training invoices only, never the held-out horizon."""
    train_ids=set(grouped.iloc[:cut]["InvoiceNo"].astype(str))
    tx=cleaned.loc[cleaned["InvoiceNo"].isin(train_ids)&cleaned["CustomerID"].notna()].copy()
    tx["CustomerID"]=tx["CustomerID"].astype(str)
    tx["revenue"]=tx["Quantity"]*tx["UnitPrice"]
    reference=pd.Timestamp(grouped.iloc[cut-1]["date"])
    agg=tx.groupby("CustomerID").agg(last=("InvoiceDate","max"),
        frequency=("InvoiceNo","nunique"),monetary=("revenue","sum"))
    if agg.empty:raise ValueError("No known customers in training")
    agg["recency"]=(reference-agg["last"]).dt.total_seconds()/86400
    # RFM ranking is trained on this prefix only. Ties get stable average ranks.
    agg["score"]=(-agg["recency"]).rank(pct=True)+agg["frequency"].rank(pct=True)+agg["monetary"].rank(pct=True)
    q1,q2=agg["score"].quantile([1/3,2/3])
    agg["segment"]=np.where(agg["score"]<=q1,"low",np.where(agg["score"]<=q2,"middle","high"))
    return agg["segment"].to_dict(),{"score_tertile_1":float(q1),"score_tertile_2":float(q2),
                                      "training_customers":len(agg)}

def invoice_customer_ids(cleaned,grouped):
    ids=cleaned.dropna(subset=["CustomerID"]).groupby("InvoiceNo")["CustomerID"].first().astype(str)
    return grouped["InvoiceNo"].map(ids).to_numpy(dtype=object)

def mine_rules(matrix,codes,support,confidence):
    freq=np.asarray(matrix.sum(axis=0)).ravel()
    active=freq>0
    if not active.any():raise ValueError("No training products")
    table=sparse_binary_frame(matrix[:,active],np.asarray(codes)[active])
    itemsets,seconds,peak=run_miner(table,"fpgrowth",support,2)
    rules=pair_rules_from_itemsets(canonical_itemsets(itemsets),codes,matrix.shape[0],confidence)
    return freq,rules,{"training_invoices":matrix.shape[0],"rules":int(rules.nnz),
                       "itemsets":len(itemsets),"seconds":seconds,"python_peak_mib":peak}

def evaluate(workbook,origins=(.5,.6,.7),horizon=.1,support=.01,confidence=.2,seed=2026):
    if not 0<support<=1 or not 0<confidence<=1:raise ValueError("Invalid mining thresholds")
    cleaned,cleaning=clean_transactions(load_transactions(Path(workbook)))
    grouped,codes,matrix=invoice_matrix(cleaned)
    customers=invoice_customer_ids(cleaned,grouped)
    folds=[]
    for fi,origin in enumerate(origins):
        cut,end=int(len(grouped)*origin),int(len(grouped)*(origin+horizon))
        if not 0<cut<end<=len(grouped):raise ValueError("Invalid chronological fold")
        segments,thresholds=rfm_training_only(cleaned,grouped,cut)
        train,test=matrix[:cut],matrix[cut:end]
        global_freq,global_rules,global_info=mine_rules(train,codes,support,confidence)
        segment_models={}
        for s in SEGMENTS:
            idx=[i for i,c in enumerate(customers[:cut]) if isinstance(c,str) and segments.get(c)==s]
            if idx:
                segment_models[s]=mine_rules(train[idx],codes,support,confidence)
        # For unseen customers and missing IDs, use global model, never a test-derived segment.
        rng=np.random.default_rng(seed+fi)
        metrics={s:{p:{"hits5":0,"hits10":0} for p in POLICIES} for s in (*SEGMENTS,"unknown")}
        counts={s:0 for s in (*SEGMENTS,"unknown")}
        co=csr_matrix((len(codes),len(codes)),dtype=np.int32)
        for i in range(test.shape[0]):
            items=set(test.indices[test.indptr[i]:test.indptr[i+1]])
            known=sorted(j for j in items if global_freq[j]>0)
            if len(known)<2:continue
            target=int(rng.choice(known))
            context=items-{target}
            cid=customers[cut+i]
            s=segments.get(cid,"unknown") if isinstance(cid,str) else "unknown"
            counts[s]+=1
            for policy in POLICIES:
                if policy=="segment_fpgrowth" and s in segment_models:
                    freq,rules,_=segment_models[s]
                else:
                    freq,rules=global_freq,global_rules
                method="popularity" if policy=="global_popularity" else "apriori"
                ranking=rank_items(context,freq,co,rules,method,10)
                metrics[s][policy]["hits5"]+=int(target in ranking[:5])
                metrics[s][policy]["hits10"]+=int(target in ranking[:10])
        results={}
        for s in (*SEGMENTS,"unknown"):
            n=counts[s]
            results[s]={"eligible":n,"policies":{
                p:{"hit_rate_5":metrics[s][p]["hits5"]/n if n else None,
                   "hit_rate_10":metrics[s][p]["hits10"]/n if n else None}
                for p in POLICIES}}
        folds.append({"origin":origin,"train_end":str(grouped.iloc[cut-1]["date"]),
                      "test_start":str(grouped.iloc[cut]["date"]),
                      "test_invoices":end-cut,"training_rfm":thresholds,
                      "global_mining":global_info,
                      "segment_mining":{s:v[2] for s,v in segment_models.items()},
                      "strata":results,"eligible":sum(counts.values())})
    return {"project":"08 FP-Growth customer RFM segment robustness",
            "source":"Official UCI Online Retail; cleaned UK positive non-cancelled invoices",
            "protocol":"Training-only customer RFM, segment-specific FP-Growth versus global FP-Growth and global popularity, identical hidden known-item targets, 50/60/70% chronological prefixes plus next 10% test windows",
            "policies":list(POLICIES),"segments":list(SEGMENTS)+["unknown"],
            "support":support,"confidence":confidence,"seed":seed,"cleaning":cleaning,
            "folds":folds,"limitations":[
                "This is within-invoice basket completion, not prediction of future purchases",
                "Segment-specific models use only training invoices from known customers",
                "RFM tertiles are fitted independently in each training prefix; no test features used",
                "Missing and new customers use the global model by design",
                "Segment-specific models may have sparse coverage and a different absolute support count",
                "Results are descriptive; no causal uplift, independent replication, or significance claim",
                "CustomerID is a retailer-specific identifier, not a demographic or fairness attribute"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("workbook",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project08_segments.json"))
    a=p.parse_args()
    result=evaluate(a.workbook)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps([{"origin":f["origin"],"eligible":f["eligible"],"strata":f["strata"]} for f in result["folds"]],indent=2))
if __name__=="__main__":main()
