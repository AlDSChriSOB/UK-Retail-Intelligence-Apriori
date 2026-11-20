# Chronological basket-completion evaluation

## Objective
Evaluate whether train-only pairwise association rules can recover one concealed product from later real UK retail invoices, relative to train-only popularity and a seeded random baseline.

## Leakage control
Sort invoices by earliest invoice timestamp, then invoice ID. Train on earliest 70%; reserve latest 30% for evaluation. Rules, popularity and catalogue are estimated only from training. Select one uniformly random *train-known* product per eligible test basket to conceal, using seed 2026. The remaining products are the observable context.

## Eligibility and metrics
Require at least two distinct train-known products in a test invoice. Report exclusions and eligible denominators. With exactly one relevant hidden product, Recall@K = HitRate@K, and Precision@K = HitRate@K / K. Report catalogue coverage as the fraction of train-known products recommended at least once. Use K=1,5,10.

## Models
Rules: directed pairs selected on training (support >=1%, confidence >=20%); score candidates by maximum confidence among observed antecedents; break ties by train popularity; use popularity fallback. Popularity: rank by training invoice frequency. Random: shuffle eligible train-known products with fixed seed. Never recommend an already observed product.

## Uncertainty
500 percentile bootstrap resamples of eligible invoices, seeded separately, estimate exploratory 95% confidence intervals for hit rate. These are *not* customer-clustered and may be too narrow when customers recur.

## Interpretation
This is retrospective basket completion using completed invoices; not a next-order forecast, A/B test, CTR, customer-personalized ranking, or causal business uplift. The 70/30 split is one historical window; no test tuning or statistical superiority claims.
