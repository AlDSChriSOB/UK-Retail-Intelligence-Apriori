"""Full reproducible audit for UCI Online Retail. Run: python -m src.audit"""
from pathlib import Path
import hashlib, json
import pandas as pd
from src.data import download_dataset, extract_dataset, load_transactions, build_basket_matrix

RAW = Path("data/raw/Online Retail.xlsx")
OUT = Path("reports/generated")
EXPECTED = ["InvoiceNo","StockCode","Description","Quantity","InvoiceDate","UnitPrice","CustomerID","Country"]

def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()

def audit(df):
    inv=df.InvoiceNo.astype("string").str.strip()
    stock=df.StockCode.astype("string").str.strip()
    desc=df.Description.astype("string").str.strip()
    country=df.Country.astype("string").str.strip()
    qty=pd.to_numeric(df.Quantity,errors="coerce")
    price=pd.to_numeric(df.UnitPrice,errors="coerce")
    cancel=inv.str.upper().str.startswith("C",na=False)
    pairs=pd.DataFrame({"i":inv,"s":stock}).dropna().groupby(["i","s"]).size()
    descs=pd.DataFrame({"s":stock,"d":desc}).dropna().groupby("s").d.nunique()
    o={"rows":len(df),"columns":len(df.columns),"column_names":list(df.columns),
       "date_min":str(df.InvoiceDate.min()),"date_max":str(df.InvoiceDate.max()),
       "unique_invoice_ids":int(inv.nunique()),"unique_stock_codes":int(stock.nunique()),
       "unique_customers":int(df.CustomerID.nunique()),"unique_countries":int(country.nunique()),
       "full_row_duplicates":int(df.duplicated().sum()),"cancellation_lines":int(cancel.sum()),
       "cancellation_invoice_ids":int(inv[cancel].nunique()),
       "negative_quantity_lines":int((qty<0).sum()),"zero_quantity_lines":int((qty==0).sum()),
       "positive_quantity_lines":int((qty>0).sum()),"negative_unit_price_lines":int((price<0).sum()),
       "zero_unit_price_lines":int((price==0).sum()),"positive_unit_price_lines":int((price>0).sum()),
       "repeated_invoice_stockcode_groups":int((pairs>1).sum()),
       "extra_repeated_invoice_stockcode_lines":int((pairs-1).clip(lower=0).sum()),
       "stockcodes_with_multiple_descriptions":int((descs>1).sum()),
       "blank_descriptions":int(desc.fillna("").eq("").sum()),
       "blank_stockcodes":int(stock.fillna("").eq("").sum()),
       "blank_countries":int(country.fillna("").eq("").sum())}
    missing=pd.DataFrame({"column":df.columns,"missing_count":[int(df[c].isna().sum()) for c in df.columns]})
    missing["missing_pct"]=missing.missing_count/max(len(df),1)*100
    numeric=df[["Quantity","UnitPrice"]].apply(pd.to_numeric,errors="coerce")
    numprof=numeric.describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
    numprof.index.name="column"
    countries=country.value_counts(dropna=False).rename_axis("country").reset_index(name="line_count")
    countries["line_pct"]=countries.line_count/max(len(df),1)*100
    monthly=df.assign(month=df.InvoiceDate.dt.to_period("M").astype(str)).groupby("month",dropna=False).size().rename("line_count").reset_index()
    return o,missing,numprof.reset_index(),countries,monthly

def clean(raw):
    x=raw.copy(); stages=[]
    def record(label,before):
        stages.append({"stage":label,"remaining_rows":len(x),"rows_removed":before-len(x),
                       "remaining_invoices":x.InvoiceNo.nunique(),"remaining_products":x.StockCode.nunique()})
    record("raw_input",len(x))
    n=len(x); x=x.dropna(subset=["InvoiceNo","StockCode","InvoiceDate"]).copy(); record("require_invoice_stockcode_date",n)
    n=len(x); c=x.InvoiceNo.astype("string").str.strip().str.upper().str.startswith("C",na=False); x=x.loc[~c].copy(); record("exclude_cancellations",n)
    n=len(x); x=x.loc[pd.to_numeric(x.Quantity,errors="coerce")>0].copy(); record("positive_quantity_only",n)
    n=len(x); x=x.loc[pd.to_numeric(x.UnitPrice,errors="coerce")>0].copy(); record("positive_unit_price_only",n)
    x["InvoiceNo"]=x.InvoiceNo.astype(str).str.strip()
    x["StockCode"]=x.StockCode.astype(str).str.strip()
    x["Country"]=x.Country.astype("string").str.strip()
    x=x.loc[x.InvoiceNo.ne("") & x.StockCode.ne("")].copy()
    all_country=x.copy()
    uk=x.loc[x.Country.str.casefold().eq("united kingdom")].copy()
    stages.append({"stage":"UK_primary_population","remaining_rows":len(uk),
                   "rows_removed":len(all_country)-len(uk),"remaining_invoices":uk.InvoiceNo.nunique(),
                   "remaining_products":uk.StockCode.nunique()})
    return uk,all_country,pd.DataFrame(stages)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    workbook=RAW if RAW.exists() else extract_dataset(download_dataset())
    raw=load_transactions(workbook)
    if list(raw.columns)!=EXPECTED: raise ValueError(f"Schema mismatch: {list(raw.columns)}")
    o,missing,numeric,countries,monthly=audit(raw)
    uk,all_country,stages=clean(raw)
    basket=build_basket_matrix(uk)
    (OUT/"raw_overview.json").write_text(json.dumps(o,indent=2),encoding="utf-8")
    missing.to_csv(OUT/"missingness.csv",index=False)
    numeric.to_csv(OUT/"numeric_profile.csv",index=False)
    countries.to_csv(OUT/"country_distribution.csv",index=False)
    monthly.to_csv(OUT/"monthly_distribution.csv",index=False)
    stages.to_csv(OUT/"cleaning_stage_counts.csv",index=False)
    uk.to_csv(OUT/"uk_clean_transactions.csv",index=False)
    all_country.to_csv(OUT/"all_country_clean_transactions.csv",index=False)
    basket.to_csv(OUT/"uk_invoice_basket.csv")
    report=f"""# UCI Online Retail: Dataset Audit and Cleaning Report

## 1. Provenance
Dataset: UCI Machine Learning Repository, Online Retail (dataset 352), created by Daqing Chen.
Citation: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33.
Licence: Creative Commons Attribution 4.0 International (CC BY 4.0).
Source workbook: {workbook.as_posix()}
SHA-256: {sha256(workbook)}
UCI describes the source period as 1 December 2010 to 9 December 2011. This is one UK-based non-store retailer, not a representative sample of all UK supermarkets or contemporary UK shoppers.

## 2. Raw data profile
| Measure | Observed value |
|---|---:|
| Rows | {o['rows']:,} |
| Columns | {o['columns']} |
| Unique invoice IDs | {o['unique_invoice_ids']:,} |
| Unique stock codes | {o['unique_stock_codes']:,} |
| Unique non-missing customers | {o['unique_customers']:,} |
| Countries | {o['unique_countries']:,} |
| Full-row duplicates | {o['full_row_duplicates']:,} |
| Cancellation lines | {o['cancellation_lines']:,} |
| Negative quantity lines | {o['negative_quantity_lines']:,} |
| Zero quantity lines | {o['zero_quantity_lines']:,} |
| Negative unit-price lines | {o['negative_unit_price_lines']:,} |
| Zero unit-price lines | {o['zero_unit_price_lines']:,} |
| Repeated invoice-stockcode groups | {o['repeated_invoice_stockcode_groups']:,} |
| Extra lines in repeated invoice-stockcode groups | {o['extra_repeated_invoice_stockcode_lines']:,} |
| Stock codes with multiple descriptions | {o['stockcodes_with_multiple_descriptions']:,} |
| Blank descriptions | {o['blank_descriptions']:,} |

Observed date range: {o['date_min']} to {o['date_max']}. Full per-column missingness is in missingness.csv; numerical distributions are in numeric_profile.csv; geography and month counts are in country_distribution.csv and monthly_distribution.csv.

## 3. Missingness and field-level treatment
Missing values are reported for every field. CustomerID is not a required field for invoice-level basket mining and missing customer IDs are retained. Customer-level analysis would require a separate missing-customer policy. Blank descriptions are profiled. StockCode is the item identifier because product descriptions can vary for a code.

## 4. Anomalies and cleaning decisions
1. Remove rows missing InvoiceNo, StockCode or InvoiceDate.
2. Treat InvoiceNo values beginning with C (case-insensitive) as cancellation invoices and exclude them from positive-sales baskets.
3. Exclude Quantity <= 0 from purchase baskets; profile negative quantities separately as return/correction candidates.
4. Exclude UnitPrice <= 0 from positive-sales baskets; profile separately.
5. Count exact duplicate rows but do not blindly delete them: repeated source lines may be legitimate. Binary basket construction ensures one product presence per invoice.
6. Report repeated invoice-stockcode pairs and stock codes with multiple descriptions. StockCode remains the stable item key.
7. Retain missing CustomerID for invoice-based analysis.
8. Preserve the cleaned all-country population; use United Kingdom as the primary analysis subset.

## 5. Sequential cleaning counts
{stages.to_markdown(index=False)}

Each removal count is relative to the immediately preceding stage. The UK filter is applied after cancellation and non-positive quantity/price filtering.

## 6. Basket construction
Transaction identifier: InvoiceNo. Item identifier: StockCode. Presence is binary: repeated lines for the same invoice and stock code collapse to one item. Primary geography is Country equal to United Kingdom after trimming and case-insensitive comparison.
- UK retained lines: {len(uk):,}
- UK invoices: {uk.InvoiceNo.nunique():,}
- UK products: {uk.StockCode.nunique():,}
- Basket dimensions: {basket.shape[0]:,} invoices x {basket.shape[1]:,} products
- Basket density: {basket.to_numpy().mean():.6f}
- Retained UK lines with missing CustomerID: {int(uk.CustomerID.isna().sum()):,}
- All-country retained positive-sales lines: {len(all_country):,}

## 7. Temporal validation and safeguards
Use chronological discovery and holdout periods. The proposed discovery cutoff is 31 August 2011; later invoices are reserved for validation. Select thresholds on discovery data only. Evaluate later-period rules against popularity and co-occurrence baselines, report threshold sensitivity and temporal stability, and do not interpret association as causation.

## 8. Generated artefacts
raw_overview.json; missingness.csv; numeric_profile.csv; country_distribution.csv; monthly_distribution.csv; cleaning_stage_counts.csv; uk_clean_transactions.csv; all_country_clean_transactions.csv; uk_invoice_basket.csv.

## 9. Limitations
Historical data from one retailer cannot establish UK-wide supermarket behaviour, current demand, or causal effects. Associations describe co-occurrence only.

## References
Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33.
UCI dataset page: https://archive.ics.uci.edu/dataset/352/online-retail
"""
    (OUT/"DATASET_AUDIT_AND_CLEANING_REPORT.md").write_text(report,encoding="utf-8")
    print(report)
    print("Generated artefacts:",OUT.resolve())

if __name__=="__main__":
    main()
