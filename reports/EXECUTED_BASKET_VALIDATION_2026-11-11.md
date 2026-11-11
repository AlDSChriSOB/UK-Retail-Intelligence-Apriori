# Independently executed basket validation — 11 November 2026

The uploaded `uk_invoice_product_basket(1).xls` is **CSV text**, containing Boolean strings `True` and `False` (not a native Excel workbook). It was scanned row by row against the complete uploaded file.

| Check | Verified |
|---|---:|
| Invoice rows | 18,019 |
| Product columns | 3,916 |
| True / present cells | 474,821 |
| Sparsity | 99.32709067% |
| Invalid binary cells | 0 |
| Empty baskets | 0 |
| Malformed row widths | 0 |
| Maximum distinct products per basket | 1,110 |
| Baskets with >100 products | 609 |
| Baskets with >250 products | 154 |

Top product codes by invoice frequency: 85123A (2,104), 85099B (1,935), 22423 (1,685), 47566 (1,593), 20725 (1,392), 84879 (1,371), 22197 (1,296), 22720 (1,241), 22457 (1,219), 20727 (1,216).

**Method:** Streaming Python CSV scan of the uploaded 404 MB export, treating `True`/`1` as present and `False`/`0` as absent. Counts and proportions were computed from all rows, not a sample.

**Research caveats:** No dates are present in the basket export; time-based evaluation needs dated transaction records. Large baskets warrant sensitivity analysis. No new Apriori rule quality metrics are claimed by this validation.

**Reproduce:** `python -m src.validate_basket "uk_invoice_product_basket(1).xls"`. The validator accepts CSV Boolean strings regardless of extension. Keep the large export local and do not commit it.
