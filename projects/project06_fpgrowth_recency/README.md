# Project 06 — FP-Growth Recency-Window Robustness

**Research question:** Does training on recent invoices improve FP-Growth basket completion compared with using all available history?

Real dataset: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), cleaned positive UK invoices. Fixed chronological training prefixes at 50%, 60%, 70% and subsequent 10% held-out windows. Training policies: expanding history, most recent 40% of training invoices, and most recent 20%. Same masked targets across policies; training-cold target rates reported.

Metrics: HR@5/10, rule counts, frequent itemsets, customer-cluster paired bootstrap intervals, training coverage and runtime. Manuscript and figures are generated from the experiment JSON. The test windows are not used to choose a winning policy.

```bash
pip install -r requirements.txt
python -m pytest -q tests/test_project06_fpgrowth_recency.py
python - <<'PY'
from src.data import download_dataset,extract_dataset
print(extract_dataset(download_dataset()))
PY
python -m projects.project06_fpgrowth_recency.analysis "data/raw/Online Retail.xlsx"
python -m projects.project06_fpgrowth_recency.paper reports/generated/project06_recency.json
```
