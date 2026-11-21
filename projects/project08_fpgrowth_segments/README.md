# Project 08 — Customer RFM Segmentation and FP-Growth Robustness

**Question:** Does customer-specific RFM segmentation improve association-rule basket completion relative to a global FP-Growth model?

Uses official [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), cleaned UK positive invoices. Three chronological origins (50%, 60%, 70%), each with the following 10% held out. RFM features (recency, unique invoices, positive revenue) and customer segment tertiles are calculated **only on training history**. Separate segment FP-Growth models and one global model are compared with a global popularity baseline on identical masked train-known items. New/missing customers receive global-model fallback.

The paper and figures are generated exclusively from measured JSON; no invented metrics or significance claims.

```bash
pip install -r requirements.txt
python -m pytest -q tests/test_project08_fpgrowth_segments.py
python - <<'PY'
from src.data import download_dataset,extract_dataset
print(extract_dataset(download_dataset()))
PY
python -m projects.project08_fpgrowth_segments.analysis "data/raw/Online Retail.xlsx"
python -m projects.project08_fpgrowth_segments.paper reports/generated/project08_segments.json
```
