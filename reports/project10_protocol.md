# Project 10: Eclat scalability and support sensitivity

Status: registered protocol, no empirical results yet.

Source: Chen (2015), Online Retail, https://doi.org/10.24432/C5BW33, CC BY 4.0. One historical UK retailer; no national representativeness claim.

Filter UK invoices, cancellations, nonpositive quantity and price; deduplicate stock codes within invoices; sort invoices by earliest timestamp with invoice identifier as tie-breaker. Use chronological prefixes 50%, 60%, 70%, 100% (100% descriptive only). Compare Eclat, Apriori and FP-Growth at support 0.005, 0.01, 0.02, 0.03, 0.05 and maximum lengths 2, 3, 4. Use identical baskets and integer minimum counts (ceiling of support times basket count). Run five timing repetitions with randomized algorithm order; separate memory measurement from timing. Record environment, CPU, package versions, source hash, cleaning counts, raw timings, medians and quartiles. Record Python allocation peaks separately from process RSS; do not conflate them. Report timeouts and failures, never silently discard. Verify exact itemset keys and integer support counts across algorithms; no equivalence claim for incomplete cells. Use small deterministic fixtures for unit tests only, not empirical results.

Acceptance: machine-readable full results, all successful cells equivalent, explicit incomplete cells, figures derived from results, reproducibility instructions, manuscript with limitations. No merge until tests and actual evidence reviewed. No customer identifiers in outputs. No causal, recommendation-accuracy or cross-country inference from computational benchmark.
