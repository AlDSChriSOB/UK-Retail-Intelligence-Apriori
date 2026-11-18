# Repeated real-UCI Apriori benchmark — 2026-11-18

**Source:** User-uploaded `Online Retail.xlsx` (541,909 original rows). Cleaned UK positive non-cancelled transactions: 485,123 rows, 18,019 invoices. Chronological train/test: 12,613/5,406 invoices. Only the **earliest N training invoices** were used.

**Method:** Independent Apriori candidate join and subset pruning, minimum support 0.05, maximum itemset length 3. Two measured runs per sample; medians below. Support values independently re-counted via transaction-ID set intersections (zero mismatches for reported frequent itemsets). This crosscheck **does not** establish absence of missing frequent itemsets.

| Invoices | Products | C2 candidates | C3 candidates | Frequent itemsets | Median mining seconds | Peak process RSS MiB |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1080 | 1711 | 92 | 207 | 0.241563 | 450.238 |
| 300 | 1773 | 990 | 220 | 294 | 0.448956 | 450.242 |
| 1000 | 2577 | 1596 | 0 | 63 | 2.087946 | 450.250 |

Frequent itemsets by size: 100 invoices: 59 / 60 / 88; 300: 45 / 82 / 167; 1000: 57 / 6 / 0.

**Memory caveat:** Process RSS includes the already-loaded original workbook and cleaned DataFrame; it is **not incremental mining memory**. RSS sampled at 5-ms intervals. Python `tracemalloc` and runtime were collected around mining. Two repeats are preliminary, not a robust performance distribution.

**Independent mlxtend real-data equivalence: BLOCKED.** `mlxtend` is not installed in the local environment and pip installation failed due to DNS/network resolution. The small fixture GitHub mlxtend test passed, but the actual UCI itemset-family equivalence has not run. Therefore PR #2 **must remain unmerged** under the user's acceptance criteria.

**Reproducibility:** This report's raw measurement JSON and standalone execution script were generated in the assistant execution environment; the source dataset is not committed to GitHub. The branch's `src/real_data_apriori_experiment.py` is the canonical script for eventual direct mlxtend equivalence.
