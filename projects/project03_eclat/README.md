# Project 03 — Eclat vs FP-Growth vs Apriori

**Dataset:** Official [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail), filtered to UK positive, non-cancelled invoices using the shared cleaning code.

**Hypothesis:** Eclat's vertical transaction-ID intersections recover the same frequent itemsets as Apriori and FP-Growth under identical support and itemset-length constraints. Under identical pair-confidence scoring, their basket-completion recommendations should be identical. Computational costs may differ.

**Experimental protocol:** Three chronological training prefixes (50%, 60%, 70%); 1% and 2% support for mining cost; maximum itemset size two for comparable recommendation rules; next 10% held-out invoice window per prefix. Conceal one training-known product per eligible invoice, then compute HR@5/10 for Eclat, Apriori, FP-Growth, cosine and popularity. Customer-clustered paired bootstrap differences use 300 resamples.

**Reproduce:**
```bash
pip install -r requirements.txt
python -m pytest -q tests/test_project03_eclat.py
python - <<'PY'
from src.data import download_dataset, extract_dataset
print(extract_dataset(download_dataset()))
PY
python -m projects.project03_eclat.mining "data/raw/Online Retail.xlsx"
python -m projects.project03_eclat.recommendation "data/raw/Online Retail.xlsx"
python -m projects.project03_eclat.paper reports/generated/project03_mining.json reports/generated/project03_recommendation.json
```

**Caveats:** Within-invoice recovery is not future purchasing or online conversion. Runtime and Python allocation measurements are exploratory, not machine-independent speed conclusions. No fabricated data or results. Figures and manuscript are generated from the actual CI artifacts; journal submission needs author declarations and editorial formatting.
