# Executed recommendation benchmark — 11 November 2026

## Source and protocol

Executed locally on the user-uploaded original UCI `Online Retail.xlsx`. Cleaning follows `src.data.clean_transactions`: positive non-cancelled UK transactions. 18,019 invoices and 3,916 product codes remain. Sort invoices by earliest timestamp; use first 12,613 invoices for training and last 5,406 for testing. Test baskets with at least two products are eligible (4,997). Hide one product uniformly at random per eligible test basket using NumPy seed 42. Recommend from all other visible products. **No test co-occurrences, item popularity or rules are used in training.**

Training-only methods:
- Popularity: rank products by training invoice frequency.
- Co-occurrence: sum training pair co-occurrence counts with visible products.
- Apriori pair rules: mine directional pairs with training support >= 1% and confidence >= 20%, then sum matching rule confidences over visible products.
- All methods exclude visible products; co-occurrence and Apriori use tiny popularity fallback when scores are zero. Ties resolve by sorted StockCode index.

## Observed held-out results

| Method | Hits@5 | Precision@5 | Recall/Hit Rate@5 | Hits@10 | Precision@10 | Recall/Hit Rate@10 |
|---|---:|---:|---:|---:|---:|---:|
| Popularity | 102 | 0.004082 | 0.020412 | 189 | 0.003782 | 0.037823 |
| Co-occurrence | 381 | 0.015249 | 0.076246 | 564 | 0.011287 | 0.112868 |
| Apriori | 435 | 0.017410 | 0.087052 | 593 | 0.011867 | 0.118671 |

Denominator: 4,997 eligible invoices. Because each invoice has **one** hidden product, recall@K and hit rate@K are mathematically identical, and precision@K = hits/(4,997 × K).

## Interpretation

Apriori pair-rule ranking outperformed the two tested baselines under this specific single-hidden-item protocol. At K=10, Apriori recovered 593 hidden items vs 564 for co-occurrence (29 additional invoices; 0.58 percentage-point absolute difference). This comparison is descriptive: no paired confidence interval or statistical test has yet been computed.

## Important limitations

1. Masking an item within an existing invoice is **basket completion**, not next-session prediction, online CTR, or business uplift.
2. The data represent one UK-based online retailer's transactions, not a representative supermarket sample.
3. One seed and one temporal split; repeated seeds, paired bootstrap intervals, and multiple time windows are needed.
4. Product variants and potentially near-duplicate StockCodes can inflate performance; inspect descriptions.
5. Customers can appear in both chronological partitions.
6. This implementation uses **two-item** directional rules, not general multi-item Apriori.
7. The code and unit tests are committed, but CI success has not been verified.

## Reproduction

`python -m src.recommendation_benchmark "data/raw/Online Retail.xlsx"`

Expected output: `reports/generated/recommendation_benchmark.json`. Do not commit the raw workbook.
