# Project 09 — Eclat-Based Frequent Itemset Mining and Comparative Market-Basket Intelligence

This project compares a transparent, vertical transaction-ID intersection implementation of **Eclat** against mlxtend Apriori and FP-Growth using genuine UK transactions from the official [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail).

Three chronological training prefixes (50%, 60%, 70%) and two support thresholds (1%, 2%) are evaluated. The strict gate requires identical itemsets and support values across all three miners before reporting exploratory runtime and Python allocation measurements. No fabricated observations, recommendation claims or causal speedup assertions.

```bash
pip install -r requirements.txt
python -m pytest -q tests/test_project09_eclat.py
python - <<'PY'
from src.data import download_dataset,extract_dataset
print(extract_dataset(download_dataset()))
PY
python -m projects.project09_eclat.analysis "data/raw/Online Retail.xlsx"
python -m projects.project09_eclat.paper reports/generated/project09_eclat.json
```
