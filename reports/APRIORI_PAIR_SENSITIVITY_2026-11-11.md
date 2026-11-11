# Executed Apriori pair-rule sensitivity analysis

**Execution date:** 2026-11-11. **Source:** user-uploaded binary invoice × StockCode basket CSV (file extension `.xls`), 18,019 invoices and 3,916 StockCodes.

## Method

Constructed a sparse invoice–item incidence matrix and computed exact pairwise co-occurrence counts. Retained unordered pairs with joint support >= 1% of invoices, then evaluated both rule directions with confidence >= 20%. Compared the full population with filters retaining invoices of at most 250 and at most 100 distinct products. Thresholds were applied separately within each population. This is **pairwise** mining, not full Apriori itemsets of lengths 3+.

| Statistic | All | At most 250 items | At most 100 items |
|---|---:|---:|---:|
| Invoices | 18,019 | 17,865 | 17,410 |
| Excluded invoices | 0 | 154 | 609 |
| Frequent singleton items | 825 | 691 | 569 |
| Frequent unordered pairs | 1,080 | 577 | 301 |
| Qualifying directional pair rules | 1,621 | 920 | 523 |
| Rule overlap with full population | — | 918 | 521 |

Compared with 1,621 full-population rules, the <=250 subset retained 918 matching rule identities and introduced 2; the <=100 subset retained 521 and introduced 2. These are rule-set overlaps, not out-of-sample precision or predictive performance.

## Example high-lift rules

- Full: `22918 → 22921`, support 0.010378, confidence 0.820175, lift 72.801681, 187 joint invoices.
- Full: `22921 → 22918`, support 0.010378, confidence 0.921182, lift 72.801681, 187 joint invoices.
- At most 250: `22916 → 22917`, support 0.010915, confidence 0.928571, lift 80.139752, 195 joint invoices.
- At most 100: `23170 → 23171`, support 0.010109, confidence 0.706827, lift 58.321628, 176 joint invoices.

## Interpretation

Excluding 609 invoices with >100 distinct products reduces qualifying pair rules from 1,621 to 523 (~67.7%). These are descriptive sensitivity results: **fewer rules do not establish better recommendation quality**. High lift can reflect complementary product variants and low marginal prevalence; product descriptions and commercial interpretation must be checked.

## Limitations and next experiments

- No dates in the basket export: chronological holdout requires the dated original transactions.
- The full notebook previously reported 3,450 association rules, potentially including longer antecedents/consequents; do not equate this with the 1,621 **two-item** rules.
- Thresholds are exploratory; do not tune them on held-out evaluation invoices.
- Next: reproduce counts in the repo, validate rule metrics against mlxtend, inspect product descriptions, and perform chronological holdout plus recommendation baselines.
