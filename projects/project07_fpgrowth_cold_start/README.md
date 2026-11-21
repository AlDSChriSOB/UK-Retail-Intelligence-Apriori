# Project 07 — FP-Growth Cold-Start and Popularity Robustness

Real-data research question: How does rule-based masked-item completion perform for unseen, rare, middle-frequency, and popular products, compared with cosine and popularity baselines?

Uses the genuine UCI Online Retail workbook and cleaned UK positive invoices. Train-only frequency tertiles, chronological 50/60/70% training prefixes, subsequent 10% heldout windows, fixed FP-Growth support 1% and confidence 20%, and a matched concealed-item task.

Unseen items are reported explicitly rather than silently excluded. All eligible multi-item invoices are evaluated, with per-stratum denominators and HR@5/10. Strata are determined exclusively by the corresponding training prefix.

```bash
pip install -r requirements.txt
python -m pytest -q tests/test_project07_fpgrowth_cold_start.py
python - <<'PY'
from src.data import download_dataset,extract_dataset
print(extract_dataset(download_dataset()))
PY
python -m projects.project07_fpgrowth_cold_start.analysis "data/raw/Online Retail.xlsx"
python -m projects.project07_fpgrowth_cold_start.paper reports/generated/project07_cold_start.json
```
