# Project 02 — FP-Growth versus Apriori on UK Online Retail

## Research question
Can FP-Growth reproduce the exact frequent one- and two-product itemsets of Apriori while reducing computational cost on **genuine historical UK retail invoices**?

## Real data and experimental design
Uses the official [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail) and the **same cleaning policy** as Project 01. Each miner is trained only on the earliest 50%, 60% or 70% of chronologically ordered invoices. Minimum support 1% and 2%; maximum itemset length two for direct comparability with Project 01's pair-rule framework. Exact itemset identities and support values must match (tolerance 1e-10); a mismatch raises an exception and fails CI.

## Reproduce
```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_project02_fpgrowth.py
python - <<'PY'
from src.data import download_dataset, extract_dataset
print(extract_dataset(download_dataset()))
PY
python -m projects.project02_fpgrowth.benchmark "data/raw/Online Retail.xlsx"
```

## Outputs
`reports/generated/project02_fpgrowth.json` contains six real-data conditions, exact itemset counts, elapsed wall-clock times, and tracemalloc peak MiB. The GitHub Actions workflow uploads this as an artifact. Do **not** report speedups until the real-data run succeeds.

## Limitations and next phase
This initial benchmark tests frequent-itemset correctness and mining cost only, **not** recommendation superiority. Timings depend on hardware and library versions; tracemalloc is not resident-set memory. Future stages: repeated timing trials, paired confidence-rule extraction, chronological basket-completion comparisons against cosine similarity and popularity, clustered bootstrap intervals, sensitivity figures and a research manuscript. This project is deliberately housed in the existing verified repository because the connected GitHub tool cannot create a new repository.
