# Executed chronological holdout — 2026-11-11

**Source:** User-uploaded UCI `Online Retail.xlsx`, 541,909 original rows. Re-executed with the repository's `src.data.clean_transactions` policy: remove missing invoice/code/date, cancellations, nonpositive quantities and prices, and non-UK rows. Result: **485,123** UK transaction rows, **18,019** invoices, **3,916** StockCodes.

## Temporal evaluation

Sort UK invoices by earliest `InvoiceDate`, then `InvoiceNo` for stable ties; first 70% train, final 30% test. Mine **directional two-item** rules only from training at support >= 1% and confidence >= 20%. Compute test co-occurrence, confidence and lift without selecting rules on the test data.

| Metric | Observed |
|---|---:|
| Training invoices | 12,613 |
| Test invoices | 5,406 |
| Last training date | 2011-09-26 11:02 |
| First test date | 2011-09-26 11:04 |
| Training directional pair rules | 1,569 |
| Rules whose antecedent occurs in test | 1,553 |
| Rules with at least one test co-occurrence | 1,546 |
| Mean training confidence across mined rules | 0.364005 |
| Mean test confidence (rules with observed antecedents) | 0.368592 |
| Median test lift (rules with observed antecedents) | 8.232295 |

## Interpretation

Most training rules had antecedents and joint occurrences in the later period. The unweighted mean confidence was comparable between the training and test periods, but this **does not establish recommendation accuracy**, statistical significance, or the absence of distribution shift. These are descriptive rule-replication diagnostics, not precision@K, recall@K, revenue uplift or causal effects.

## Limitations and safeguards

- Evaluation is **invoice-chronological**, not a customer-disjoint split; customers can recur.
- The full basket dataset has no dates; this experiment uses the original dated workbook instead.
- Two-item rules only; no claim of full multi-item Apriori validation.
- Confidence is undefined when the antecedent never appears in test; these 16 rules are excluded from the test-confidence average, not counted as zero.
- Rules are equally weighted in the reported mean; volume-weighted estimates may differ.
- No test-period hyperparameter tuning was performed.
- Next: implement top-K recommendation ranking and evaluate against popularity/co-occurrence baselines with an explicit within-invoice masking protocol, confidence intervals and time-slice robustness.

## Reproduction

`python -m src.chronological_holdout "data/raw/Online Retail.xlsx"`

This report records results executed locally against the supplied workbook; CI/test success is not claimed.
