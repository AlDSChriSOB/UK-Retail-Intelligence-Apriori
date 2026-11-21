# Project 04 — Temporal Stability of UK Retail Association Rules

**Question:** Do rules discovered on earlier UK retail invoices maintain confidence and support in later invoices?

This project uses the **genuine UCI Online Retail dataset** and shared cleaning code. It mines directional product-pair rules at 50%, 60%, and 70% chronological training prefixes and evaluates them on the next 10% of invoices without using held-out data to select rules.

Metrics include conditional confidence replication, support-and-confidence retention, missing-antecedent counts, mean paired confidence drift, and exploratory rule-level bootstrap intervals. Two figures and a scientific manuscript are generated from measured JSON.

**Reproduce:**
```bash
pip install -r requirements.txt
python -m pytest -q tests/test_project04_rule_drift.py
python - <<'PY'
from src.data import download_dataset,extract_dataset
print(extract_dataset(download_dataset()))
PY
python -m projects.project04_rule_drift.analysis "data/raw/Online Retail.xlsx"
python -m projects.project04_rule_drift.paper reports/generated/project04_drift.json
```

**Interpretation:** Rules share products and invoice histories, so rule-level bootstrap intervals are descriptive, not independent-sampling inference. Rule stability is not causal concept drift or online conversion impact. No synthetic research data are used.
