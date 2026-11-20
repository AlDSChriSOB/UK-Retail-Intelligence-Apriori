# Multi-origin paired customer-cluster evaluation

## Pre-specified protocol
This analysis complements the original 70/30 single holdout with three expanding-window origins at 50%, 60% and 70% of chronological UK invoices. Each origin is evaluated on the immediately following 10% of invoices. These test windows are non-overlapping. All rule counts, training catalogue and popularity scores are recomputed from that fold's training history only.

## Identical test targets across models
Each eligible test invoice has at least two distinct products known in the fold's training catalogue. One train-known product is randomly concealed using a fixed seed per fold. The three models receive identical visible baskets and hidden targets. K=1,5,10. Hit rate equals recall because there is exactly one concealed relevant product. The primary comparisons are paired invoice-level hit differences (rules minus popularity and rules minus random).

## Dependence-aware intervals
For each comparison, invoices are grouped by the source CustomerID. Customer clusters, rather than individual invoices, are sampled with replacement 500 times. Each bootstrap replicate recomputes the difference in hit rate across all sampled invoices. A 2.5th/97.5th percentile interval is reported. Missing CustomerID invoices receive distinct invoice-specific cluster labels; thus unidentified repeat customers cannot be accounted for. Do not interpret interval exclusion of zero as multiplicity-adjusted significance.

## Scope and remaining limitations
Three adjacent test windows are more robust than one split but are not external replications. Rule thresholds remain fixed (minimum support 1%, confidence 20%) and are not tuned on test data. Randomly concealing an item from an already completed invoice evaluates basket completion, not chronological next-item purchase. Eligibility excludes cold-start items. This is not an online A/B experiment and cannot estimate causal revenue uplift. Confidence intervals are descriptive exploratory intervals with no multiplicity correction.

## Reproduction
Run `python -m src.temporal_clustered "data/raw/Online Retail.xlsx" --bootstrap 500`. CI saves `reports/generated/temporal_clustered.json` as an evidence artifact. Only report numerical values after this workflow has completed successfully.
