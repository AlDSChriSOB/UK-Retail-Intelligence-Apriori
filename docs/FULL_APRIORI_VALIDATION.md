# Full Apriori validation (Task 1)

This independent level-wise implementation performs L1 counting, candidate self-joins, downward-closure pruning, higher-order itemset counting and association-rule generation. It is separate from the original pairwise recommendation benchmark, which is unchanged.

## Local validation

Run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pip install mlxtend
python -m pytest -q tests/test_full_apriori.py
```

The mlxtend test compares the complete frequent-itemset family and numerical support, confidence and lift on a deterministic five-basket fixture. The test skips if mlxtend is not installed. For full validation it **must run, not skip**.

## Data and interpretation

This implementation consumes iterables of string product codes, with duplicate codes within a basket deduplicated. Do not fit frequent itemsets on held-out invoices. Do not interpret association as causality. The current implementation is educational/reference-grade, with straightforward candidate counting; on the full UCI dataset high-dimensional candidate enumeration may be expensive. Benchmark memory/runtime before full-catalogue experiments, and compare on a predeclared manageable training subset if needed.

## Acceptance criteria

- Candidate pruning verified
- Frequent 3-itemset verified
- Support, confidence and lift verified
- Independent mlxtend equivalence test passes without skipping
- Original benchmark and its published results remain unchanged
- No claims of full UCI execution without actual execution evidence

References: Agrawal & Srikant (1994), *Fast Algorithms for Mining Association Rules in Large Databases*; mlxtend Apriori and association-rules documentation.
