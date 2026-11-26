import sys, os, json, math, itertools, subprocess, threading, time, statistics, hashlib, platform, importlib.metadata, random
from pathlib import Path
import psutil
import tracemalloc
from src.project10 import mine
from src.data import download_dataset, extract_dataset, load_transactions, clean_transactions

def measure(fn, mode):
    if mode == 'python':
        tracemalloc.start()
        try:
            result = fn()
            current, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        return result, dict(peak_python_bytes=peak)
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    stop = threading.Event()
    def sample():
        while not stop.wait(.002):
            samples.append(process.memory_info().rss)
    thread = threading.Thread(target=sample)
    thread.start()
    try:
        result = fn()
        samples.append(process.memory_info().rss)
    finally:
        stop.set()
        thread.join()
    return result, dict(baseline_rss_bytes=baseline, sampled_peak_rss_bytes=max(samples), delta_rss_bytes=max(samples)-baseline, samples=len(samples))

def worker():
    path, name, n, threshold, length, mode = sys.argv[2:]
    baskets = json.loads(Path(path).read_text())[:int(n)]
    result, metrics = measure(lambda: mine(name,baskets,int(threshold),int(length)),mode)
    digest = hashlib.sha256(json.dumps(sorted(result.items()),separators=(',',':')).encode()).hexdigest()
    print(json.dumps(dict(digest=digest,count=len(result),**metrics)))

def main():
    out=Path('reports/generated/project10_memory'); out.mkdir(parents=True,exist_ok=True)
    def allocation():
        x=bytearray(32*1024*1024)
        for i in range(0,len(x),4096): x[i]=1
        time.sleep(.05)
        return x
    controls={}
    for mode in ['rss','python']:
        obj, metrics=measure(allocation,mode)
        controls[mode]=metrics
        del obj
    assert controls['rss']['delta_rss_bytes']>=24*1024*1024, controls
    assert controls['python']['peak_python_bytes']>=32*1024*1024, controls
    workbook=extract_dataset(download_dataset())
    clean,audit=clean_transactions(load_transactions(workbook))
    baskets=clean.groupby('InvoiceNo').agg(date=('InvoiceDate','min'),items=('StockCode',lambda x:sorted(set(x)))).reset_index().sort_values(['date','InvoiceNo'])['items'].tolist()
    path=out/'baskets.json'; path.write_text(json.dumps(baskets))
    evidence=dict(controls=controls,source_sha256=hashlib.sha256(workbook.read_bytes()).hexdigest(),platform=platform.platform(),cpu=Path('/proc/cpuinfo').read_text().split('model name')[1].splitlines()[0].strip(': \t'),python=sys.version,packages={p:importlib.metadata.version(p) for p in ['psutil','numpy','pandas','mlxtend','scipy']},rows=[])
    rng=random.Random(2026)
    try:
        for fraction,support,length in itertools.product([.5,.6,.7,1],[.005,.01,.02,.03,.05],[2,3,4]):
            n=int(len(baskets)*fraction); threshold=math.ceil(n*support)
            for mode in ['rss','python']:
                for rep in range(3):
                    names=['eclat','apriori','fpgrowth']; rng.shuffle(names)
                    for name in names:
                        row=dict(fraction=fraction,support=support,length=length,mode=mode,rep=rep,algorithm=name)
                        try:
                            p=subprocess.run([sys.executable,'-m','src.project10_memory','worker',str(path),name,str(n),str(threshold),str(length),mode],capture_output=True,text=True,timeout=180,check=True)
                            row.update(json.loads(p.stdout.strip().splitlines()[-1]))
                        except (subprocess.SubprocessError,ValueError) as exc:
                            row['error']=str(exc)
                        evidence['rows'].append(row)
                        (out/'evidence.json').write_text(json.dumps(evidence,indent=2))
    finally:
        path.unlink(missing_ok=True)
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    df=pd.DataFrame(evidence['rows'])
    assert len(df)==1080 and 'error' not in df.columns
    keys=['fraction','support','length']
    assert (df.groupby(keys).digest.nunique()==1).all()
    assert (df.groupby(keys+['mode','algorithm']).size()==3).all()
    for mode,metric in [('rss','delta_rss_bytes'),('python','peak_python_bytes')]:
        sub=df[df['mode']==mode]
        summary=sub.groupby(keys+['algorithm'])[metric].agg(median='median',q1=lambda x:x.quantile(.25),q3=lambda x:x.quantile(.75)).reset_index()
        summary.to_csv(out/(mode+'.csv'),index=False)
        fig,axes=plt.subplots(2,2,figsize=(10,7),sharex=True)
        for ax,(fraction,g) in zip(axes.flat,summary[summary.length==2].groupby('fraction')):
            for name,h in g.groupby('algorithm'):
                ax.plot(h.support*100,h['median']/1048576,marker='o',label=name)
            ax.set(title=f'Prefix {fraction:.0%}',xlabel='Support (%)',ylabel='MiB')
        axes.flat[0].legend(); fig.tight_layout()
        fig.savefig(out/(mode+'.png'),dpi=300); fig.savefig(out/(mode+'.pdf')); plt.close(fig)

if __name__=='__main__':
    worker() if len(sys.argv)>1 and sys.argv[1]=='worker' else main()
