# Reproducibility protocol — Project 01

**Status:** Versioned protocol for public-data reruns. The research uses genuine UCI Online Retail records, not synthetic experimental data.

## Data and provenance
Dataset: Chen, D. (2015), *Online Retail*, UCI Machine Learning Repository, DOI: 10.24432/C5BW33, https://archive.ics.uci.edu/dataset/352/online+retail . Original transactions cover December 2010–December 2011. The GitHub Actions workflow downloads the official archive at runtime and records SHA-256 of the extracted workbook. The repository does not redistribute raw customer-linked records.

## Environment and commands
Python 3.11 on Ubuntu GitHub Actions; dependencies from `requirements.txt`. For a local rerun:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q tests/test_robust_benchmark.py
python - <<'PY'
from src.data import download_dataset, extract_dataset
print(extract_dataset(download_dataset()))
PY
python -m src.robust_benchmark "data/raw/Online Retail.xlsx" --bootstrap 300
python -m src.paper_figures reports/generated/robust_benchmark.json
```

On Windows, activate with `.venv\\Scripts\\activate`. To record the installed environment: `python -m pip freeze > environment-freeze.txt`. To record the raw workbook checksum: `sha256sum "data/raw/Online Retail.xlsx"` (or use Python hashlib on Windows).

## Pre-specified experimental grid
Three expanding chronological training origins: 50%, 60%, 70%; following 10% test windows. Nine support/confidence combinations: support 0.005/0.01/0.02 and confidence 0.10/0.20/0.30. K=5,10. Seed 2026, bootstrap 300 resamples. Models: summed-confidence directional pair rules, raw co-occurrence, binary cosine similarity, and popularity. The same masked target is used across models and grid configurations within each fold. Evaluation is train-catalogue-eligible within-invoice recovery, not future-purchase prediction.

## Audit and artifact map
- `reports/generated/robust_benchmark.json`: complete per-window threshold grid and paired customer-cluster percentile intervals
- `reports/generated/paper_figures/figure_1_method_comparison.pdf`: fixed-setting HR@10 by model and window
- `reports/generated/paper_figures/figure_2_threshold_sensitivity.pdf`: full Apriori threshold grid
- `reports/generated/paper_figures/supplementary_grid.csv`: machine-readable expanded comparison table
- `reports/environment-freeze.txt`: exact installed package versions for that run
- `reports/uci-workbook-sha256.txt`: SHA-256 of source workbook

The workflow uploads these files as a downloadable artifact. The prior verified run is https://github.com/AlDSChriSOB/UK-Retail-Intelligence-Apriori/actions/runs/44745182299 ; this new figure-generation workflow requires a fresh successful run.

## Interpretation and limitations
Exploratory bootstrap intervals are not multiplicity adjusted. CustomerIDs may be missing; repeat customers may cross windows. The dataset is historical, product-code catalogue mappings may vary, and results do not establish online uplift or causal effects. The method called Apriori is **pairwise confidence rule scoring**, not a full higher-order itemset recommender. Never merge single 70/30 and rolling 10%-window results into one estimate.

## Submission audit checklist
- [ ] Confirm author list, affiliations, corresponding email and ORCID with the author
- [ ] Choose a target journal and apply its exact manuscript template and reference style
- [ ] Verify latest workflow and examine artifact plots for legibility and exact numeric agreement
- [ ] Freeze submission code to a Git tag or release with commit SHA
- [ ] Check citations and claims against originals
- [ ] Confirm funding, competing interests, ethics/data governance and AI-assistance declarations
- [ ] Perform final editorial review and plagiarism/originality checks
