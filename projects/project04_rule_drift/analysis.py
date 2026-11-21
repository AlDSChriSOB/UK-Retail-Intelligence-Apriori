"""Project 04: out-of-time rule-confidence drift and stability on real UK invoices."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from src.data import clean_transactions,load_transactions
from src.chronological_holdout import invoice_matrix,rule_statistics

def confidence_for_rules(matrix, pairs):
    """Conditional confidence P(B|A), undefined when antecedent never appears."""
    x=matrix.tocsr()
    counts=np.asarray(x.sum(axis=0)).ravel()
    co=(x.astype(np.int64).T@x.astype(np.int64)).tocsr()
    return np.asarray([float(co[a,b])/counts[a] if counts[a]>0 else np.nan
                       for a,b in pairs],dtype=float)

def paired_bootstrap_confidence_drift(train_conf,test_conf,replicates=1000,seed=2026):
    """Paired rule-level bootstrap; descriptive uncertainty, rules not independent."""
    a=np.asarray(train_conf,float);b=np.asarray(test_conf,float)
    keep=np.isfinite(a)&np.isfinite(b)
    diff=b[keep]-a[keep]
    if not len(diff) or replicates<1:
        raise ValueError("No comparable rules or invalid replicates")
    rng=np.random.default_rng(seed)
    means=np.empty(replicates)
    for i in range(replicates):
        means[i]=diff[rng.integers(0,len(diff),len(diff))].mean()
    return {"mean_change":float(diff.mean()),"median_abs_change":float(np.median(abs(diff))),
            "mean_change_rule_bootstrap_95pct_ci":list(map(float,np.quantile(means,[.025,.975]))),
            "n_rules":int(len(diff))}

def evaluate(workbook,origins=(.5,.6,.7),horizon=.1,min_support=.01,
             min_confidence=.2,bootstrap=1000,seed=2026):
    cleaned,cleaning=clean_transactions(load_transactions(Path(workbook)))
    grouped,codes,matrix=invoice_matrix(cleaned)
    n=matrix.shape[0]
    folds=[]
    for fi,origin in enumerate(origins):
        cut,end=int(n*origin),int(n*(origin+horizon))
        if not 0<cut<end<=n:
            raise ValueError("Invalid chronological window")
        train,test=matrix[:cut],matrix[cut:end]
        rules,counts=rule_statistics(train,codes,min_support,min_confidence)
        pairs=[(a,b) for a,b,_ in rules]
        if not pairs:
            raise ValueError("No rules in training window")
        train_conf=np.asarray([joint/counts[a] for a,b,joint in rules],float)
        test_conf=confidence_for_rules(test,pairs)
        observed=np.isfinite(test_conf)
        if not observed.any():
            raise ValueError("No test antecedents observed")
        change=paired_bootstrap_confidence_drift(train_conf,test_conf,bootstrap,seed+fi)
        retention=float(np.mean(test_conf[observed]>=min_confidence))
        test_counts=np.asarray(test.sum(axis=0)).ravel()
        co=(test.astype(np.int64).T@test.astype(np.int64)).tocsr()
        test_support=np.asarray([float(co[a,b])/test.shape[0] for a,b in pairs])
        joint_retention=float(np.mean((test_conf[observed]>=min_confidence)&(test_support[observed]>=min_support)))
        changes=sorted(({"antecedent":str(codes[a]),"consequent":str(codes[b]),
                         "train_confidence":float(train_conf[i]),"test_confidence":float(test_conf[i]),
                         "delta":float(test_conf[i]-train_conf[i])}
                        for i,(a,b) in enumerate(pairs) if observed[i]),key=lambda z:abs(z["delta"]),reverse=True)
        folds.append({"origin":origin,"train_invoices":cut,"test_invoices":end-cut,
                      "train_end":str(grouped.iloc[cut-1]["date"]),
                      "test_start":str(grouped.iloc[cut]["date"]),
                      "train_rules":len(rules),"observed_antecedent_rules":int(observed.sum()),
                      "unobserved_antecedent_rules":int((~observed).sum()),
                      "test_confidence_retention_rate":retention,
                      "test_support_and_confidence_retention_rate":joint_retention,
                      "drift":change,"largest_absolute_confidence_changes":changes[:20]})
    return {"project":"04 UK retail association-rule temporal stability",
            "source":"Official UCI Online Retail; positive UK invoices",
            "protocol":"Training-only directional rules at 50/60/70% prefixes; following 10% chronological holdout; fixed 1% support/20% confidence",
            "bootstrap":bootstrap,"seed":seed,"cleaning":cleaning,"folds":folds,
            "limitations":["Rule-level bootstrap treats selected rules as sampling units despite overlapping products and correlated rules; descriptive only",
                           "No multiple-testing adjusted rule significance","Three neighboring test windows are not independent external replications",
                           "Confidence drift is not proof of causal concept drift",
                           "Only frequent directional pairs; no next-order recommendation accuracy claimed"]}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("workbook",type=Path)
    p.add_argument("--output",type=Path,default=Path("reports/generated/project04_drift.json"))
    a=p.parse_args()
    result=evaluate(a.workbook)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps([{k:v for k,v in f.items() if k not in ("largest_absolute_confidence_changes",)}
                      for f in result["folds"]],indent=2))
if __name__=="__main__":
    main()
