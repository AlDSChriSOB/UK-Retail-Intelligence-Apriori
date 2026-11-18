# Measured independent Apriori results — 18 November 2026

Source: user-uploaded `Online Retail.xlsx` (23 MB), original 541,909 rows.
The repository cleaning policy retained 485,123 UK positive, non-cancelled sales rows and 18,019 invoices. Chronological 70/30 split: 12,613 training invoices and 5,406 test invoices. The samples below use **only the earliest training invoices**, never test invoices.

| Training invoices | Sample products | Frequent 1-itemsets | Frequent 2-itemsets | Frequent 3-itemsets | Total frequent itemsets | Apriori seconds | Python-tracked peak MiB |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 1,080 | 59 | 60 | 88 | 207 | 0.25781 | 0.24401 |
| 300 | 1,773 | 45 | 82 | 167 | 294 | 0.43422 | 0.10784 |
| 1,000 | 2,577 | 57 | 6 | 0 | 63 | 1.81414 | 0.22348 |

Parameters: min_support=0.05; max_len=3. Wall times use `time.perf_counter()` around the independent Apriori call only, excluding workbook loading, cleaning, and basket construction. Memory is Python `tracemalloc` peak during mining, **not total process RSS**. Results are from one run each; do not interpret differences as scaling estimates or memory efficiency claims. Itemset counts vary with sampled invoice composition.

**Independent comparison status:** `mlxtend` was unavailable in the local execution environment and installation failed due to network resolution. Therefore these real-data outputs have **not** yet been independently checked against mlxtend; the separate small-fixture GitHub CI equivalence tests did pass. The committed experiment script is intended to perform real-data equivalence in a dependency-equipped environment.

The data were processed locally from the uploaded original workbook. These results are observational descriptive computations, not recommendation accuracy or causal findings.
