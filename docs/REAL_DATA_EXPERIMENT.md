# Real UCI Apriori experiment

This is a **separate, bounded experiment** from the existing basket-completion benchmark. No real-data runtime, memory, or itemset counts are claimed until the script is executed successfully.

## Run

```bash
python -m pip install -r requirements.txt
python -m src.data
python -m src.real_data_apriori_experiment "data/raw/Online Retail.xlsx" --max-invoices 1000 --min-support 0.05 --max-len 3
```

The script uses the earliest 1,000 invoices from the chronological training portion of cleaned UK transactions (or fewer if unavailable). The later 30% is never used for mining. It selects only products present in this sample, executes the independent Apriori and mlxtend with matching thresholds, and fails if itemsets or supports disagree. Output is written to `reports/generated/full_apriori_real_data.json`.

**Resource protection:** Start with 100 invoices if necessary; increase gradually only after observing runtime and memory. The independent reference implementation uses exhaustive candidate scans and can become expensive on high-cardinality retail baskets. `tracemalloc` measures Python-tracked peak allocations, **not** total RSS. Run on a dedicated machine and record overall RSS separately for a publication-quality resource comparison.

**Acceptance:** Original data obtained from UCI; sample size and provenance recorded; both algorithms complete; exact itemset families match; supports agree to 1e-10; measured runtime and tracked peak memory recorded; no synthetic data or held-out invoices included.

**Ethics:** Historic one-retailer transactions cannot support contemporary UK supermarket generalisation. Do not publish customer identifiers or raw transactional rows.
