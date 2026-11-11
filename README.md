# UK Retail Intelligence — Association Rules and Basket Completion

[![Python tests](https://github.com/AlDSChriSOB/UK-Retail-Intelligence-Apriori/actions/workflows/python-tests.yml/badge.svg)](https://github.com/AlDSChriSOB/UK-Retail-Intelligence-Apriori/actions/workflows/python-tests.yml)

**Research-oriented data science portfolio project:** discovering product associations and evaluating historical basket-completion recommendations against transparent baselines.

> **Dataset scope:** UCI Online Retail (December 2010–December 2011) is one UK-based online retailer with international customers. It is **not** a contemporary or representative UK supermarket dataset.

## Research question

Do directional two-item association rules learned from earlier transactions recover withheld products in later invoices more effectively than popularity and simple co-occurrence baselines? How sensitive are conclusions to chronological windows, masking seeds and candidate catalogue duplicates?

## Main findings (historical benchmark)

| Method | Hit Rate@5 | Hit Rate@10 |
|---|---:|---:|
| Popularity | 2.04% | 3.78% |
| Co-occurrence | 7.62% | 11.29% |
| Directional pair rules | **8.71%** | **11.87%** |

- Chronological 70/30 invoice split: **12,613** training invoices; **5,406** test invoices; **4,997** test invoices with at least two distinct products.
- One seeded product hidden per eligible test basket; remaining products form the query.
- At K=10, pair rules exceeded co-occurrence by **0.58 percentage points**. The previously reported paired 95% bootstrap interval was **−0.18 to +1.32 percentage points**, so **superiority over co-occurrence is not established**.
- Temporal 60/40 and 80/20 comparisons and five masking seeds are documented in [research reports](reports/). Collapsing 129 shared-description candidate groups (265 codes) into representatives changed performance only modestly for the two checked seeds.
- This is **basket completion**, not future-session prediction, causal uplift or a live recommendation service.

### Results visualisation

The notebook generates `results/figures/hit_rate_by_method.png` from actual benchmark outputs; the image is not hard-coded into this repository.

## Explore the project

| Component | Where |
|---|---|
| Research notebook | [notebooks/02_recommendation_research.ipynb](notebooks/02_recommendation_research.ipynb) |
| Interactive demo | [app.py](app.py) |
| Benchmark implementation | [src/recommendation_benchmark.py](src/recommendation_benchmark.py) |
| Paired bootstrap | [src/paired_bootstrap.py](src/paired_bootstrap.py) |
| Catalogue sensitivity | [src/catalogue_sensitivity.py](src/catalogue_sensitivity.py) |
| Research reports | [reports/](reports/) |
| CI workflow | [.github/workflows/python-tests.yml](.github/workflows/python-tests.yml) |

## Reproduce

Use Python 3.11 or later:

```bash
python -m pip install -r requirements.txt
python -m src.data
python -m pytest -q
python -m src.recommendation_benchmark "data/raw/Online Retail.xlsx"
python -m src.paired_bootstrap reports/generated/per_invoice_hits.csv
python -m src.catalogue_sensitivity "data/raw/Online Retail.xlsx"
streamlit run app.py
```

The dataset downloader retrieves the [official UCI Online Retail archive](https://archive.ics.uci.edu/dataset/352/online+retail). Raw data are not committed. The notebook can be opened from the repository or notebooks directory. Running it writes charts and results locally.

### Research design

Cleaning removes cancellations, nonpositive quantities and prices, and non-UK transactions; missing CustomerID is retained for invoice-level analysis. The first 70% of invoices sorted by earliest invoice timestamp are used for training. Popularity uses training product frequencies, co-occurrence sums training pair counts, and the association-rule model sums confidences of training-derived directional pairs (minimum support 1%, confidence 20%). All exclude visible basket items. A tiny popularity fallback is used for candidates without association scores. No test invoices are used to fit these models.

**Terminology:** The recommender implemented here mines **two-item directional association rules**. It should not be described as an exhaustive general multi-item Apriori implementation without a separate implementation and validation.

### Limitations and interpretation

One historical retailer, overlapping customers across time, a single hidden item per invoice, possible product variants, and overlapping temporal sensitivity windows limit generalisation. Hit Rate@K equals Recall@K under the single-hidden-item protocol. Association is not causation. Passing GitHub Actions confirms configured tests run successfully, not that empirical research claims have been independently peer-reviewed.

## Attribution

Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository. [DOI:10.24432/C5BW33](https://doi.org/10.24432/C5BW33). Source dataset licensed CC BY 4.0.
