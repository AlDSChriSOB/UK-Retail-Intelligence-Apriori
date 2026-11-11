# Temporal and product-description robustness audit — 11 November 2026

**Execution:** Original user-supplied `Online Retail.xlsx`, UK positive non-cancelled invoices; invoice-level chronology; single randomly hidden product per test basket (seed 42). Train-only pair rules require support >=1% and confidence >=20%. The comparison is descriptive, not a claim of statistical superiority.

| Training fraction | Training invoices | Test invoices | Eligible test baskets | Popularity Hit@10 | Co-occurrence Hit@10 | Apriori Hit@10 | Apriori minus co-occurrence |
|---|---:|---:|---:|---:|---:|---:|---:|
| 60% | 10,811 | 7,208 | 6,667 | 3.630% (242) | 11.069% (738) | 11.699% (780) | +0.630 pp |
| 70% | 12,613 | 5,406 | 4,997 | 3.782% (189) | 11.287% (564) | 11.867% (593) | +0.580 pp |
| 80% | 14,415 | 3,604 | 3,347 | 3.376% (113) | 11.712% (392) | 12.160% (407) | +0.448 pp |

The 60% training split ends 2011-08-18 12:06 and test begins 2011-08-18 12:23. The 80% training split ends 2011-10-26 12:14 and test begins 2011-10-26 12:15. All results are from the same historical retailer and share overlapping train/test populations; these are not independent replications. A 95% paired bootstrap CI for the 70/30 split at K=10 was previously reported as −0.18 to +1.32 percentage points; it crosses zero.

## Product-description collisions

Across the 3,916 UK StockCodes, a conservative audit chose each code's **most frequent normalized nonempty product description** (uppercase, trimmed, whitespace-collapsed), then grouped codes sharing an identical selected description. Results:

- **129** distinct normalized descriptions shared by at least two codes.
- **265** StockCodes belong to these shared-description groups.

This identifies possible catalog duplication or code reuse, **not proof that products are physically identical**. Color/size variants can have different descriptions and are not captured by this exact-match screen. The next experiment should collapse or exclude confirmed duplicate product groups and recompute rankings before interpreting product recommendation gains.

## Testing and CI verification

Six local **research-artifact integrity tests** passed (temporal counts, metric arithmetic, hit bounds, variant counts, seed outcome shape, seed hit count agreement): `6 passed in 0.09s`. These are **not the repository's full pytest suite**. GitHub's commit workflow lookup was limited to PR-triggered runs and returned none; the combined commit status lookup returned no statuses. Consequently, the **GitHub Actions workflow remains unverified**; do not label it green.

## Limitations

- All comparisons use basket completion, not a future purchase prediction.
- Temporal windows overlap and all use seed 42.
- Identical descriptions are only candidate duplicate products; no impact-on-Hit@10 test has yet been executed.
- No paired confidence intervals for the 60/40 and 80/20 windows have been computed.
- This is a two-item rule recommender, not full multi-item Apriori.
