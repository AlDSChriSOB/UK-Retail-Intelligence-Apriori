# Temporal robustness of association-rule basket completion in UK online retail: a reproducible comparative study

**Manuscript status:** Journal-neutral IMRaD submission candidate; final journal-specific layout, author metadata and declarations pending. Earlier numerical results were verified from successful GitHub Actions runs; the expanded comparator and sensitivity grid completed successfully on GitHub Actions run 44745182299, with artifact 14633825538 independently inspected. Results remain exploratory and should not be interpreted as online uplift.

## Abstract
**Background:** Association rules are interpretable but may fail to generalize across changing retail demand. **Objective:** Assess whether training-only directional pair rules recover a concealed item from later retail baskets more reliably than simple baselines, and evaluate temporal stability. **Methods:** The public UCI Online Retail transaction workbook was restricted to positive, non-cancelled UK invoices and converted into invoice–product binary incidence data. The primary experiment trained on the first 70% of chronological invoices and evaluated a randomly concealed train-known product from eligible later invoices. Three additional non-overlapping temporal windows were evaluated using expanding 50%, 60% and 70% training histories and subsequent 10% test windows. Hit Rate@K, paired hit-rate differences and customer-cluster bootstrap intervals were computed. **Results:** In the initial 70/30 test, 4,904 eligible invoices yielded Hit Rate@10 of 13.17% for association rules, 3.53% for popularity and 0.29% for a seeded random baseline. Across three additional temporal windows (n=1,630; 1,658; 1,620 eligible invoices), rule-based Hit Rate@10 was 18.04%, 17.73% and 13.83%, versus popularity at 6.38%, 4.70% and 3.21%. All three exploratory customer-clustered 95% bootstrap intervals for the paired difference in Hit Rate@10 were positive. **Conclusions:** Directional pair rules outperformed the tested simple baselines in retrospective within-invoice basket completion, but these results do not establish future-purchase prediction, online conversion lift, or external validity. In the expanded benchmark, cosine similarity outperformed summed-confidence Apriori at the pre-specified 1% support and 20% confidence setting in all three windows (HR@10: 21.96%, 21.71%, 21.17% versus 16.87%, 17.13%, 14.20%). Apriori performance varied considerably with thresholds. Thus the interpretable rule approach was not the best tested method.

**Keywords:** association-rule mining; Apriori; market basket; temporal evaluation; recommendation; clustered bootstrap; reproducibility

## 1. Introduction
Market-basket association rules quantify frequent product co-purchases and offer interpretable recommendation candidates. Retrospective random train/test partitions can overstate generalization when transaction patterns shift. This study therefore asks: (RQ1) can rules trained on earlier UK invoices recover concealed products in later invoices? (RQ2) does their advantage over popularity persist across distinct historical windows? (RQ3) how sensitive is performance to minimum-support and minimum-confidence thresholds, and how does it compare with item co-occurrence and cosine-normalized item similarity? The contribution is an openly reproducible evaluation pipeline, not a claim of commercial deployment.

## 2. Related work and conceptual background
Association-rule mining formalizes support and confidence for co-occurring itemsets. For antecedent A and consequent B, support(A,B) is the proportion of training invoices containing both; confidence(A→B) is the proportion containing B among invoices containing A. Rule confidence need not represent causal dependence and is influenced by item prevalence. Popularity ranking is a strong diagnostic baseline for non-personalized retail recommendation. Co-occurrence counts and cosine-normalized binary item similarity offer complementary item-item baselines without minimum-support filtering. Offline basket completion differs fundamentally from predicting a customer's next transaction.

**Source citations:** The study draws on the original Apriori publication [2], the UCI dataset [1] and the foundational bootstrap reference [3]. Bibliographic metadata below have been checked against the dataset repository, VLDB archive and journal record.

## 3. Materials and methods

### 3.1 Data provenance and cleaning
Data: UCI Machine Learning Repository, Online Retail (dataset 352) [1], downloaded from the official UCI archive at https://archive.ics.uci.edu/dataset/352/online+retail . The cleaning code drops rows with missing invoice, stock code or timestamp, excludes invoice numbers beginning C, non-positive quantities and prices, and non-UK transactions. Repeated invoice-product lines are collapsed to binary presence. Products are identified by StockCode; this may retain catalogue synonyms and stock-code inconsistencies. CustomerID may be missing.

### 3.2 Chronological splitting
Invoices are ordered by minimum InvoiceDate and InvoiceNo. The original primary analysis uses a 70/30 chronological split. Three additional expanding-window analyses use train fractions 0.50, 0.60 and 0.70 and the next 10% of invoices as each test window. All models are re-estimated on the training portion of each fold. Adjacent test windows are disjoint; training histories overlap by design.

### 3.3 Basket-completion target and eligibility
Each test invoice must contain at least two distinct products present in the training catalogue. Exactly one such product is selected with a fixed pseudo-random seed and concealed. The remaining products form the query context, identical for all methods within an invoice. Thus Hit Rate@K equals Recall@K and Precision@K equals Hit Rate@K divided by K. Excluded invoices are counted; cold-start performance is not estimated. The target is not the customer's next purchase.

### 3.4 Recommendation algorithms
**Association rules:** Directional pairwise rules are learned on training invoices only. The earlier primary experiment ranks candidates using maximum confidence over observed antecedents with popularity fallback. **Popularity:** Rank products by frequency in training invoices. **Random:** Seeded random ranking of train-known unobserved products. **Expanded benchmark (verified real-data results):** For each observed product, sum training-only pair-rule confidences; compare with summed raw co-occurrence and summed cosine-normalized binary item similarities. All models exclude observed and training-unseen products, with deterministic popularity fallback and product-code tie-breaking. Because the expanded rule aggregator differs from the earlier maximum-confidence implementation, its estimates must not be described as a direct replication of the earlier rule model.

### 3.5 Sensitivity analysis
The expanded benchmark fixes minimum-support values at 0.005, 0.01 and 0.02 and minimum-confidence values at 0.10, 0.20 and 0.30. Nine settings are evaluated in each of three temporal windows at K=5 and K=10, without selecting an optimal setting using test data. Sensitivity comparisons are exploratory and should be reported as a grid, not cherry-picked maxima.

### 3.6 Paired uncertainty
Per-invoice hit indicators for all methods share the same concealed target. For each pairwise model comparison, the mean difference in hit indicators estimates the difference in Hit Rate@K. The earlier temporal analysis uses 500 percentile bootstrap replicates, resampling known CustomerID clusters within each fold; invoices with missing CustomerID receive invoice-specific clusters. The expanded sensitivity grid uses 300 replicates per configuration. Intervals are exploratory, not multiplicity-adjusted, and unknown repeat customers can remain dependent.

### 3.7 Computational reproducibility
Python implementation: src/heldout_topk.py, src/temporal_clustered.py, src/robust_benchmark.py. GitHub Actions workflow definitions: .github/workflows/heldout-topk.yml, temporal-clustered.yml and robust-benchmark.yml. Unit tests exercise ranking exclusions and deterministic cluster-bootstrap behavior. The UCI workbook is downloaded during CI; generated JSON and figures are archived as workflow artifacts. Pseudorandom seeds and thresholds are recorded in output. The submission evidence workflow archives an exact pip-freeze snapshot and workbook SHA-256, alongside figure PDFs/PNGs and the machine-readable sensitivity grid. See docs/REPRODUCIBILITY.md. The source requirements remain minimum-version specifications; the captured environment snapshot identifies the actual CI environment.

## 4. Results (verified prior experiments)

### 4.1 Single chronological 70/30 holdout
The real-data workflow produced 4,904 eligible test invoices, 3,726 training-catalogue products and 1,569 qualifying directional pair rules.

| Model | HR@1 | HR@5 | HR@10 |
|---|---:|---:|---:|
| Association rules | 5.00% | 10.32% | 13.17% |
| Popularity | 0.59% | 2.08% | 3.53% |
| Random | 0.04% | 0.16% | 0.29% |

The invoice-bootstrap 95% percentile interval for rule HR@10 was 12.20%–14.20%. This interval is not customer-clustered.

### 4.2 Three chronological windows
| Training origin | Eligible test invoices | Rules HR@10 | Popularity HR@10 | Paired difference (percentage points) | Customer-clustered 95% CI for difference (pp) |
|---|---:|---:|---:|---:|---|
| 50% | 1,630 | 18.04% | 6.38% | +11.66 | [9.84, 13.63] |
| 60% | 1,658 | 17.73% | 4.70% | +13.03 | [11.36, 15.00] |
| 70% | 1,620 | 13.83% | 3.21% | +10.62 | [8.98, 12.23] |

All intervals exclude zero under this exploratory bootstrap protocol. Because multiple windows and K values are inspected, these should not be promoted to confirmatory family-wise significance claims.

### 4.3 Expanded comparator and threshold-grid results (verified)

The successful real-UCI Actions run 44745182299 evaluated all nine support/confidence combinations on each of three temporal windows, using the same concealed target across algorithms within each fold. The primary fixed setting below is minimum support 0.01 and confidence 0.20. These results use the **summed-confidence** Apriori scorer, not the earlier maximum-confidence scorer, and therefore are not numerically interchangeable with Sections 4.1–4.2.

| Method, HR@10 | 50% origin (n=1,630) | 60% origin (n=1,658) | 70% origin (n=1,620) |
|---|---:|---:|---:|
| Cosine item similarity | **21.96%** | **21.71%** | **21.17%** |
| Apriori summed confidence | 16.87% | 17.13% | 14.20% |
| Raw item co-occurrence | 13.93% | 15.74% | 13.58% |
| Popularity | 5.52% | 4.64% | 3.95% |

The paired customer-cluster bootstrap 95% intervals for **Apriori minus cosine** HR@10 differences (percentage points) were [-7.09, -3.19], [-6.81, -2.62] and [-9.11, -4.53] for the three origins. Each interval was negative. For **Apriori minus raw co-occurrence**, corresponding intervals were [1.53, 4.40], [0.09, 2.88] and [-0.65, 1.90], so the third window did not demonstrate a clear advantage over co-occurrence.

Across the nine fixed threshold settings, summed-confidence Apriori HR@10 ranged from **10.43% to 21.66%** (first period), **10.55% to 21.77%** (second) and **8.70% to 18.83%** (third). These are descriptive within-grid minima and maxima, **not** held-out optimized estimates. Higher minimum support of 0.02 generally reduced HR@10 in these experiments. This sensitivity undermines any claim that a single threshold is universally robust.

All figures in this section were extracted from the archived JSON artifact `robust-real-uci-benchmark` (artifact ID 14633825538). The reported bootstrap intervals are unadjusted for multiplicity and involve only 300 cluster resamples per configuration.

## 5. Discussion
The positive temporal paired differences suggest that co-purchase structure carries information beyond global item popularity for within-invoice reconstruction. Rule performance declines in the third temporal window, consistent with possible distributional variation, although this experiment does not identify the cause. The interpretable pair-rule representation is attractive for analysis but may omit higher-order complementarity and customer-specific preferences. The executed expanded comparisons show that cosine-normalized item similarity performed better than summed-confidence Apriori in all three windows. This is a substantive negative finding for claims of Apriori superiority and suggests that normalizing item co-occurrence by item frequency matters for this task.

## 6. Limitations and threats to validity
1. **Task mismatch:** Masking an item within a completed basket does not simulate a recommendation served before checkout or predict a future invoice.
2. **Selection bias:** Evaluation excludes invoices without at least two training-known items; new products and sparse baskets are underrepresented.
3. **Temporal generalizability:** Three neighboring 2011 windows are not independent market replications or a contemporary deployment test.
4. **Customer dependence:** Known CustomerID is clustered within folds, but missing IDs are invoice-specific, potentially understating uncertainty.
5. **Multiple comparisons:** Threshold-grid intervals are not adjusted for multiplicity; no confirmatory superiority p-value is claimed.
6. **Baseline breadth:** Current comparators omit matrix factorization, neural recommenders and sequential session models. Those require careful fit-for-purpose adaptation.
7. **Catalogue integrity:** StockCode aliases and description collisions can alter co-occurrence patterns.
8. **Algorithm definition:** The initial maximum-confidence rule scorer differs from the expanded summed-confidence scorer.
9. **Reproducibility:** Official public data and CI scripts are available; the submission workflow records the workbook SHA-256 and pip-freeze environment. These files should be checked in the successful artifact before submission.
10. **Causal inference:** No online randomized experiment, conversion measure, profit measure or business uplift estimate is available.

## 7. Conclusions
Using real historical UK invoices and chronological evaluation, training-only directional pair rules consistently exceeded global popularity for the narrowly defined task of recovering one hidden item. The result is promising evidence for interpretable basket completion, not evidence of deployment effectiveness. The completed sensitivity grid shows substantial dependence on support/confidence thresholds, and cosine similarity is the strongest tested method at the fixed 1%/20% rule setting. These results qualify the earlier conclusions and favor reporting Apriori as an interpretable, but not performance-leading, baseline.

## Data availability, code availability and declarations
**Data:** Public UCI Online Retail dataset 352; raw data not redistributed in the repository. **Code:** https://github.com/AlDSChriSOB/UK-Retail-Intelligence-Apriori . **Evidence:** GitHub Actions run 44285881737 (single holdout), 44744099669 (temporal clustered), and expanded comparator run 44745182299 (artifact 14633825538). **Ethics:** Secondary analysis of public transaction data; do not release individual-level CustomerID-linked outcomes. **Funding, conflicts, author contributions:** Author must confirm before submission; not assumed. **Generative AI disclosure:** Drafting assistance was used; the author must verify scientific claims and comply with target-journal policies.

## References
1. Chen, D. (2015). *Online Retail* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33
2. Agrawal, R., & Srikant, R. (1994). Fast algorithms for mining association rules in large databases. In *Proceedings of the 20th International Conference on Very Large Data Bases* (pp. 487–499). Morgan Kaufmann. https://www.vldb.org/conf/1994/P487.PDF
3. Efron, B. (1979). Bootstrap methods: Another look at the jackknife. *The Annals of Statistics, 7*(1), 1–26. https://doi.org/10.1214/aos/1176344552

## Figure legends and supplementary material
**Figure 1.** Chronological-window Hit Rate@10 for cosine item similarity, summed-confidence directional pair rules, raw co-occurrence and popularity. Fixed rule setting: support 0.01, confidence 0.20. Error bars are not shown; paired clustered intervals are reported in the text.

**Figure 2.** Directional pair-rule Hit Rate@10 across all nine minimum-support and minimum-confidence combinations, separately for the three chronological windows. Cells show percentage hit rates. This is exploratory sensitivity analysis, not test-set hyperparameter optimization.

**Supplementary Table S1.** Full window × threshold × K × method performance grid, including paired rule-minus-comparator bootstrap intervals where applicable, in machine-readable CSV format.

**Figure generation:** `python -m src.paper_figures reports/generated/robust_benchmark.json`. Figure PDF and PNG files are produced by the submission-evidence workflow and should be checked for legibility before final submission.

## Journal submission formatting and required author metadata
This draft follows a **journal-neutral IMRaD structure**, with a structured abstract, numbered sections, tables, numbered references, figure legends, reproducibility and declarations. It is **not** represented as compliant with a particular journal's word limits, abstract structure, artwork specifications or reference style until a target journal is selected. Author-supplied details still required: final author order, affiliation(s), corresponding author, ORCID (if any), funding, competing interests, ethics statement as applicable, acknowledgments and journal-required AI-use disclosure. No journal acceptance or submission is claimed.
