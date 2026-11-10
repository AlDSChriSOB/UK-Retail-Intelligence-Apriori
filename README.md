# UK Retail Intelligence using Apriori

**Research project:** Discovering and evaluating product association rules in historical UK retail transactions.

**Status:** Research scaffold. No empirical findings are claimed until the UCI data pipeline and holdout evaluation have been executed.

## Research question

Can Apriori identify meaningful product associations in historical transactions from a UK-based online retailer, and do rules discovered in an earlier period generalise to later transactions?

## Dataset

- Source: [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- DOI: https://doi.org/10.24432/C5BW33
- Citation: Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository.
- Coverage: 1 December 2010 to 9 December 2011; 541,909 source records.
- The source is one UK-based non-store retailer, not a representative sample of all UK supermarkets or present-day UK consumers.
- The dataset contains transactions from multiple countries. The primary analysis filters Country == "United Kingdom".

The dataset is licensed CC BY 4.0. Attribute the dataset creator and UCI. Raw data is downloaded from UCI and is not committed to this repository.

## Research workflow

1. Download and audit the original workbook.
2. Apply transparent cleaning rules and report row counts.
3. Construct invoice-level binary baskets using StockCode.
4. Mine frequent itemsets with an independently implemented Apriori algorithm.
5. Validate itemsets and support against mlxtend.
6. Evaluate support/confidence thresholds, rule redundancy and stability.
7. Discover rules in an earlier period and evaluate them on a later chronological holdout.
8. Compare masked-item recommendation performance against popularity and co-occurrence baselines.

## Primary cleaning policy

- Require non-missing invoice number, stock code and invoice date.
- Exclude cancellation invoices whose InvoiceNo begins with C.
- Exclude non-positive quantities and unit prices from the primary sales-basket analysis; profile these records separately.
- Filter to United Kingdom.
- Retain missing CustomerID because baskets are invoice-based.
- Collapse repeated invoice/product rows to binary presence.
- Use StockCode as the item identifier.

These are analytical decisions, not universal rules. Report the number of rows removed at every stage.

## Repository structure

- configs/: analysis configuration
- data/raw/: downloaded source data (git-ignored)
- notebooks/: end-to-end exploratory and baseline analysis
- results/tables/ and results/figures/: generated outputs (git-ignored by default)
- src/: acquisition, cleaning, Apriori and validation code
- tests/: deterministic unit tests

## Dataset audit and cleaning report

After installing dependencies, run:

```bash
python -m src.audit
```

This downloads the official UCI archive when the workbook is not already present in data/raw/, audits provenance/schema/missingness/duplicates/cancellations/returns/prices/countries/dates/product-code consistency, applies the documented positive-sales cleaning policy, and builds UK invoice-level binary baskets. It writes the measured Markdown report and CSV/JSON artefacts to reports/generated/. The raw workbook remains local and is not committed.

Missing CustomerID values are retained for invoice-level basket analysis. Cancellation invoices, non-positive quantities and non-positive unit prices are profiled and excluded from the positive-sales basket population. Exact duplicate rows are reported but not blindly removed; binary invoice-item construction handles repeated item lines transparently.

## Setup

Use Python 3.10 or later. Create and activate a virtual environment, then install dependencies with pip install -r requirements.txt. Run python -m src.data from the repository root to download, audit, clean and construct the basket matrix. Open notebooks/01_apriori_retail_analysis.ipynb and run cells in order.

## Reproducibility and research safeguards

- The discovery cutoff is initially 31 August 2011; the later period is reserved for evaluation.
- Select thresholds using discovery data only. Do not tune on the final holdout.
- Validate implementation outputs against mlxtend and hand-computed examples.
- Record package versions, thresholds, cleaning counts and random seeds.
- Do not claim that association implies causation.

## Acceptance criteria

- [ ] Dataset acquired and provenance recorded.
- [ ] Audit and cleaning counts saved.
- [ ] Basket construction documented and tested.
- [ ] Apriori unit tests pass.
- [ ] Frequent itemsets match mlxtend on controlled and real data.
- [ ] Rule metrics independently checked.
- [ ] Chronological holdout evaluation completed without leakage.
- [ ] Popularity/co-occurrence baselines evaluated.
- [ ] Threshold sensitivity and temporal stability assessed.
- [ ] Figures, tables and limitations documented.

## Limitations

This is a historical single-retailer dataset. It cannot support claims about all UK supermarkets, current UK-wide purchasing behaviour, household affordability or causal effects of product placement. Association rules describe co-occurrence, not causation.
