# Catalogue-duplication sensitivity and CI verification audit — 11 November 2026

## Executed experiment
Dataset: original UCI Online Retail workbook, cleaned UK non-cancelled positive-price and positive-quantity invoices. For each StockCode, choose its most frequent nonempty description after uppercasing, trimming and collapsing whitespace. Group codes whose primary descriptions match exactly; use lexicographically smallest code as representative. Collapse duplicate invoice-product entries, then **retrain** popularity, co-occurrence and directional pair-rule models separately for each catalogue. Evaluate same 70/30 invoice-chronological split, with one hidden item per eligible test invoice; fixed seeds 42 and 7. Thresholds: support >=1%, confidence >=20%.

- Original StockCodes: **3,916**.
- Candidate shared-description groups: **129**, containing **265** StockCodes.
- Collapsed catalogue: **3,780** representative StockCodes (136 fewer).
- Training invoices: **12,613**; test invoices: **5,406**; eligible test baskets: **4,997** in both catalogues.

### Hit Rate@10 (hits / 4,997)

| Seed | Catalogue | Popularity | Co-occurrence | Apriori pair rules |
|---|---|---:|---:|---:|
| 42 | Original | 189 (3.782%) | 564 (11.287%) | 593 (11.867%) |
| 42 | Collapsed | 188 (3.762%) | 562 (11.247%) | 587 (11.747%) |
| 7 | Original | 168 (3.362%) | 547 (10.947%) | 535 (10.706%) |
| 7 | Collapsed | 170 (3.402%) | 544 (10.887%) | 536 (10.726%) |

Collapsing exact-description candidate duplicates changes Apriori Hit Rate@10 by **−0.120 percentage points** for seed 42 and **+0.020 percentage points** for seed 7. The relative ordering of Apriori and co-occurrence is unchanged in both seeds. These are descriptive sensitivities, not an inference that all variants are harmless or that Apriori is statistically superior.

## Local verification

A separate local integrity test module was executed against the saved experiment outputs and sparse matrices: **5 passed in 0.17s**. Tests cover population sizes, catalogue reduction, rates versus integer hit counts, Hit@5 <= Hit@10, and reproduction of original published seed-42 counts. This verifies saved experiment integrity; it is **not** the repository's complete pytest suite.

## GitHub Actions CI status — not verified

- CI workflow file: `.github/workflows/python-tests.yml`, committed previously.
- Latest commit checked: `193e54f4323815b3959d7c341f93855635799214`.
- GitHub connector's **combined commit statuses** endpoint returned `statuses: []` for latest and workflow-creation commits.
- GitHub connector's **commit workflow runs** endpoint returned `workflow_runs: []` for both, but its implementation only searches pull-request-triggered runs; it cannot establish push-triggered GitHub Actions status.
- Direct public GitHub Actions page retrieval was unavailable from the execution environment.
- Therefore **full CI success or failure cannot honestly be certified**. Do not represent the workflow as passing. Open the repository Actions tab and inspect the latest `Python tests` run, or provide its URL/log for a definitive verdict.

## Reproduction
`python -m src.catalogue_sensitivity "data/raw/Online Retail.xlsx"`

No raw workbook is committed. The code and its catalogue mapping regression tests are tracked in `src/catalogue_sensitivity.py` and `tests/test_catalogue_sensitivity.py`.

## Research limitations
Exact primary-description equality can merge separate products and miss true near-duplicates; customer-disjoint and rolling-origin evaluations remain to be done. This is invoice basket completion, not next-session purchase prediction. The same 4,997 invoices were evaluated at two seeds, so seeds are not independent datasets.
