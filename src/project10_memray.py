import json, math, hashlib, itertools, tempfile
from pathlib import Path
import memray
from src.project10 import mine
from src.data import download_dataset, extract_dataset, load_transactions, clean_transactions

def profile(fn, path):
    with memray.Tracker(str(path), native_traces=True, trace_python_allocators=True):
        result=fn()
    reader=memray.FileReader(str(path))
    peak=sum(r.size for r in reader.get_high_watermark_allocation_records())
    return result, peak

def main():
    out=Path('reports/generated/project10_memray'); out.mkdir(parents=True,exist_ok=True)
    workbook=extract_dataset(download_dataset())
    clean,_=clean_transactions(load_transactions(workbook))
    baskets=clean.groupby('InvoiceNo').agg(date=('InvoiceDate','min'),items=('StockCode',lambda x:sorted(set(x)))).reset_index().sort_values(['date','InvoiceNo'])['items'].tolist()
    evidence={'source_sha256':hashlib.sha256(workbook.read_bytes()).hexdigest(),'memray_version':memray.__version__,'rows':[]}
    with tempfile.TemporaryDirectory() as tmp:
        path=Path(tmp)/'control.bin'
        obj,peak=profile(lambda:bytearray(32*1024*1024),path)
        assert peak>=32*1024*1024
        del obj
        evidence['control_peak_bytes']=peak
        for i,(fraction,support,length) in enumerate(itertools.product([.5,1],[.005,.05],[2,4])):
            subset=baskets[:int(len(baskets)*fraction)]
            threshold=math.ceil(len(subset)*support)
            digests=[]
            for name in ['eclat','apriori','fpgrowth']:
                path=Path(tmp)/f'{i}-{name}.bin'
                result,peak=profile(lambda:mine(name,subset,threshold,length),path)
                digest=hashlib.sha256(json.dumps(sorted(result.items()),separators=(',',':')).encode()).hexdigest()
                digests.append(digest)
                evidence['rows'].append(dict(fraction=fraction,support=support,length=length,algorithm=name,peak_tracked_bytes=peak,digest=digest,count=len(result)))
                (out/'evidence.json').write_text(json.dumps(evidence,indent=2))
                path.unlink()
                del result
            assert len(set(digests))==1
    assert len(evidence['rows'])==24
if __name__=='__main__':
    main()
