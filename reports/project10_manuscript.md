# Computational scalability and support sensitivity of frequent itemset mining in historical UK online retail
## Abstract
We compared Eclat, low-memory Apriori and FP-Growth across 60 configurations of chronological basket prefixes, support thresholds and maximum itemset lengths, with five repetitions per algorithm (900 runs). All configurations produced matching canonical itemset/support digests. Eclat had the lowest median runtime in 50 configurations, Apriori in ten. Median paired configuration speed ratios (Apriori/Eclat and FP-Growth/Eclat) were 2.886 and 4.371. All recorded peak RSS values were identical; memory efficiency is therefore unresolved. Results concern these implementations and one historical retailer.
## Introduction and related work
Frequent itemsets underpin association-rule discovery. Apriori exploits downward closure; FP-Growth compresses transactions into a prefix tree; Eclat intersects vertical transaction identifier sets. These contrasting representations motivate an empirical comparison rather than a universal complexity ranking.
## Methods
UCI Online Retail (Chen, 2015), CC BY 4.0, contains 541,909 rows. Existing cleaning removed cancellations, nonpositive quantity/price and non-UK transactions, retaining 485,123 rows, 18,019 invoices and 3,916 stock codes. Invoices were deduplicated by stock code and ordered by earliest timestamp and invoice identifier. Prefixes 50/60/70/100%, support 0.5/1/2/3/5%, maximum lengths 2/3/4; integer support threshold ceiling(n*s). Five repetitions in randomized algorithm order per cell, Linux subprocesses, 120-second timeout. Timing includes algorithm-specific representation construction, excludes workbook loading and digest serialization. Apriori and FP-Growth use mlxtend; Eclat uses Python sets. Canonical sorted itemsets and integer counts were hashed with SHA-256. Compare medians within configurations and summarize paired ratios descriptively; overlapping prefixes and repeated thresholds are not independent samples, so no population-level significance test is claimed.
## Results
All 60 cells and 900 runs present; 180 CSV rows reconciled against JSON medians, counts and RSS. Each cell has five repetitions per algorithm and matching digests/counts. Itemset counts range 33–23,974.
| Algorithm | Min median (s) | Median of medians (s) | Max median (s) | Median within-cell IQR (s) | Wins |
|---|---:|---:|---:|---:|---:|
| Eclat | 0.049 | 0.298 | 6.263 | 0.0036 | 50 |
| Apriori | 0.719 | 0.856 | 5.793 | 0.0125 | 10 |
| FP-Growth | 0.783 | 1.318 | 7.883 | 0.0170 | 0 |
Paired configuration ratios Apriori/Eclat median 2.886 (Q1 1.123, Q3 6.373); FP-Growth/Eclat 4.371 (1.808, 7.899). At 0.5% support Apriori wins 7/12; at 1% it wins 3/12; Eclat wins all 36 cells at support >=2%. These quartiles describe configuration heterogeneity, not confidence intervals.
## Memory audit and figure review
Every raw RSS observation equals 620888 KiB (606.34 MiB). The process lifetime high-water mark may be dominated by inherited or startup memory; the cause is not established. Reject this endpoint for comparative inference. The runtime PNG was visually inspected and is legible; it depicts only the 70% prefix, length two slice. The RSS PNG was inspected and shows overlapping constant curves; retain solely as diagnostic evidence. A separate measurement run with current RSS sampled during mining and baseline RSS recorded is needed before any memory claim.
## Discussion
Eclat's advantage at higher support is consistent with pruning reducing intersection work, but this mechanism was not isolated experimentally. Apriori can be faster at low support. Implementation language, sparse encoding, machine load and catalogue structure confound algorithm-level interpretation. No predictive recommendation accuracy, causal uplift, or cross-country inference follows.
## Limitations and reproducibility
One historical retailer; nested prefixes; five repeats; one Linux environment; no independent replication; CPU and dependency versions not fully recorded; digest agreement is not an independent exhaustive oracle. Memory endpoint invalid. Reproduce with pip install -r requirements.txt; python -m pytest tests/test_project10.py -q; python -m src.project10. Evidence: GitHub Actions run 45835398377, artifact 15127299313, commit b2b740505723822cea6a49d88f3140330ac48f01. Source SHA256 43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d. No raw customer identifiers published. No funding or conflict declarations verified; author to confirm before submission.
## Conclusion
Runtime and digest agreement are supported by the archived evidence. Memory comparison remains unresolved; this is a manuscript draft, not a submitted or peer-reviewed paper.
## References
Chen, D. (2015). Online Retail. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33.
Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules. VLDB.
Han, J., Pei, J., & Yin, Y. (2000). Mining frequent patterns without candidate generation. SIGMOD.
Zaki, M. J. (2000). Scalable algorithms for association mining. IEEE Transactions on Knowledge and Data Engineering, 12(3), 372–390.
Bibliographic details of algorithm references require final independent verification before submission.
