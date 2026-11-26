import math, time, random, json, hashlib, platform, subprocess, sys, itertools, statistics, resource
from pathlib import Path
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, fpgrowth
from src.data import download_dataset, extract_dataset, load_transactions, clean_transactions

def eclat(baskets, threshold, length):
    vertical = {}
    for i, basket in enumerate(baskets):
        for item in basket:
            vertical.setdefault(item, set()).add(i)
    out = {}
    def visit(prefix, candidates):
        for i, (item, tids) in enumerate(candidates):
            key = prefix + (item,)
            out[key] = len(tids)
            if len(key) < length:
                tail = [(j, tids & other) for j, other in candidates[i+1:]]
                visit(key, [(j,t) for j,t in tail if len(t)>=threshold])
    visit((), [(i,t) for i,t in sorted(vertical.items()) if len(t)>=threshold])
    return out

def mine(name, baskets, threshold, length):
    if name == 'eclat':
        return eclat(baskets, threshold, length)
    from mlxtend.preprocessing import TransactionEncoder
    enc = TransactionEncoder()
    matrix = enc.fit(baskets).transform(baskets, sparse=True)
    frame = pd.DataFrame.sparse.from_spmatrix(matrix, columns=enc.columns_)
    fn = apriori if name=='apriori' else fpgrowth
    result = fn(frame, min_support=(threshold-0.25)/len(baskets), use_colnames=True, max_len=length, **({'low_memory':True} if name=='apriori' else {}))
    return {tuple(sorted(k)):round(s*len(baskets)) for k,s in zip(result.itemsets,result.support)}

def worker():
    path,name,n,threshold,length = sys.argv[2:]
    baskets=json.loads(Path(path).read_text())[:int(n)]
    start=time.perf_counter()
    result=mine(name,baskets,int(threshold),int(length))
    elapsed=time.perf_counter()-start
    canonical=json.dumps(sorted(result.items()),separators=(',',':'))
    print(json.dumps(dict(seconds=elapsed, count=len(result), digest=hashlib.sha256(canonical.encode()).hexdigest(), peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))

def main():
    out=Path('reports/generated/project10'); out.mkdir(parents=True,exist_ok=True)
    workbook=extract_dataset(download_dataset())
    raw=load_transactions(workbook); clean,audit=clean_transactions(raw)
    grouped=clean.groupby('InvoiceNo').agg(date=('InvoiceDate','min'),items=('StockCode',lambda x:sorted(set(x)))).reset_index().sort_values(['date','InvoiceNo'])
    baskets=grouped['items'].tolist()
    path=out/'temporary_baskets.json'; path.write_text(json.dumps(baskets))
    evidence=dict(audit=audit,source_sha256=hashlib.sha256(workbook.read_bytes()).hexdigest(),environment=platform.platform(),python=sys.version,rows=[])
    rng=random.Random(2026)
    try:
        for fraction,support,length in itertools.product([.5,.6,.7,1],[.005,.01,.02,.03,.05],[2,3,4]):
            n=int(len(baskets)*fraction); threshold=math.ceil(n*support)
            cell=dict(fraction=fraction,support=support,length=length,n=n,threshold=threshold,runs=[])
            for rep in range(5):
                names=['eclat','apriori','fpgrowth']; rng.shuffle(names)
                for name in names:
                    try:
                        p=subprocess.run([sys.executable,'-m','src.project10','worker',str(path),name,str(n),str(threshold),str(length)],capture_output=True,text=True,timeout=120,check=True)
                        row=json.loads(p.stdout.strip().splitlines()[-1])
                    except (subprocess.SubprocessError,ValueError) as exc:
                        row={'error':type(exc).__name__}
                    cell['runs'].append(dict(algorithm=name,rep=rep,**row))
                if any('error' in x for x in cell['runs']): break
            cell['verified']=len(cell['runs'])==15 and all('digest' in x for x in cell['runs']) and len({x.get('digest') for x in cell['runs']})==1
            evidence['rows'].append(cell)
            (out/'evidence.json').write_text(json.dumps(evidence,indent=2))
    finally:
        path.unlink(missing_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records=[]
    for cell in evidence['rows']:
        if cell['verified']:
            for name in ['eclat','apriori','fpgrowth']:
                runs=[x for x in cell['runs'] if x['algorithm']==name]
                records.append(dict(fraction=cell['fraction'],support=cell['support'],length=cell['length'],algorithm=name,median_seconds=statistics.median(x['seconds'] for x in runs),median_rss=statistics.median(x['peak_rss_kib'] for x in runs),count=runs[0]['count']))
    frame=pd.DataFrame(records); frame.to_csv(out/'summary.csv',index=False)
    if not frame.empty:
        for metric in ['median_seconds','median_rss']:
            fig,ax=plt.subplots()
            subset=frame[(frame.fraction==.7)&(frame.length==2)]
            for name,g in subset.groupby('algorithm'):
                ax.plot(g.support,g[metric],marker='o',label=name)
            ax.set(xlabel='Minimum support',ylabel=metric); ax.legend()
            fig.savefig(out/(metric+'.png'),dpi=300); fig.savefig(out/(metric+'.pdf')); plt.close(fig)
    assert all(c['verified'] for c in evidence['rows']), 'Incomplete or unequal cells; inspect evidence'

if __name__=='__main__':
    worker() if len(sys.argv)>1 and sys.argv[1]=='worker' else main()
