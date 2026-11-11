# Paired bootstrap and masking-seed robustness — executed 11 November 2026

## Data and evaluation
Recomputed from the user-supplied UCI Online Retail workbook after repository cleaning: 18,019 UK invoices, 12,613 chronological training invoices, 5,406 test invoices, of which 4,997 have at least two distinct products. One hidden item per eligible invoice, same trained models, deterministic seeds. Apriori means **directional two-item association rules** with training support >=1% and confidence >=20%, scored by summed confidence with a popularity fallback.

## Paired 95% bootstrap confidence intervals (masking seed 42)
10,000 **paired invoice-level** resamples, bootstrap seed 20261111. Difference is Apriori minus baseline in **percentage points**.

| Comparison | Hit-rate difference | 95% percentile bootstrap CI | Apriori-only hits | Baseline-only hits |
|---|---:|---:|---:|---:|
| Apriori – co-occurrence, @5 | +1.081 pp | +0.440 to +1.741 pp | 162 | 108 |
| Apriori – popularity, @5 | +6.664 pp | +5.924 to +7.424 pp | 357 | 24 |
| Apriori – co-occurrence, @10 | +0.580 pp | **−0.180 to +1.321 pp** | 199 | 170 |
| Apriori – popularity, @10 | +8.085 pp | +7.284 to +8.925 pp | 438 | 34 |

At @10 the Apriori-vs-co-occurrence interval crosses zero: **no statistically established superiority at the conventional two-sided 5% level** using this interval. At @5 the corresponding interval excludes zero for this single split and seed, but robustness across masking seeds is still required before generalisation.

## Multiple masking seeds (same chronological train/test split)

| Seed | Popularity Hit@10 | Co-occurrence Hit@10 | Apriori Hit@10 | Apriori minus co-occurrence |
|---|---:|---:|---:|---:|
| 42 | 3.782% | 11.287% | 11.867% | +0.580 pp |
| 7 | 3.362% | 10.947% | 10.706% | −0.240 pp |
| 19 | 3.562% | 11.067% | 11.247% | +0.180 pp |
| 123 | 3.402% | 10.967% | 11.127% | +0.160 pp |
| 2026 | 3.762% | 11.067% | 11.427% | +0.360 pp |

The Apriori-vs-co-occurrence advantage reverses for seed 7. Five seeds are a sensitivity check, **not five independent datasets**; do not treat them as independent replicates for a p-value.

## Research conclusion
Basket-completion recommendation performance from co-occurrence and Apriori substantially exceeds the popularity baseline under this protocol. Apriori's advantage over co-occurrence is small and unstable; claiming robust superiority at @10 would be unsupported.

## Reproducibility and CI
The numerical calculations were executed locally against the original uploaded workbook, with paired outcomes and results archived in conversation-generated JSON. The repository has `src/recommendation_benchmark.py`, `src/paired_bootstrap.py`, unit tests and `.github/workflows/python-tests.yml`. **A passing GitHub Actions run has not been confirmed.** The GitHub connector's commit workflow query returned no PR-triggered runs, and the combined commit status endpoint returned no statuses; neither establishes a successful or failed push-triggered workflow.

## Limitations
- Basket completion (single hidden item) is not next-visit recommendation.
- One chronological split; customer overlap and product variants are unaddressed.
- No adjustment for testing multiple metrics and seeds.
- Statistical inference is conditional on the observed invoice population, model specification and one masking seed.
- Reported CI is a percentile paired bootstrap, not a formal preregistered hypothesis test.
