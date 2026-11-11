# Basket validation and executed-notebook findings

**Project:** UK Retail Market-Basket Intelligence using Apriori  
**Evidence status:** Results below were reported from the uploaded executed notebook and independently inspected exported basket in the project conversation. They have **not** been independently reproduced by running the complete raw-data pipeline in CI.

## Dataset and notebook results

| Metric | Value | Evidence |
|---|---:|---|
| Raw transaction rows | 541,909 | Executed notebook |
| Cleaned UK transaction rows | 485,123 | Executed notebook |
| UK invoices / basket rows | 18,019 | Notebook and exported basket |
| UK product codes / basket columns | 3,916 | Notebook and exported basket |
| Discovery invoices | 11,327 | Executed notebook |
| Later evaluation invoices | 6,692 | Executed notebook |
| Frequent itemsets | 2,252 | Executed notebook |
| Association rules | 3,450 | Executed notebook |
| Basket sparsity | 99.3271% | Notebook, consistent with exported basket |

## Exported basket checks

The uploaded `uk_invoice_product_basket(1).xls` was found to be **CSV-formatted text with an .xls extension**, not a native Excel workbook. Independent checks reported:

- 18,019 invoice rows and 3,916 product columns
- 0 duplicate invoice identifiers; 0 duplicate product headers
- 0 malformed row widths; 0 invalid binary entries; 0 empty baskets
- 474,821 product-presence entries; 26.35 distinct products per basket on average
- Maximum basket size: **1,110** distinct products

### Most frequent StockCode values

| StockCode | Invoices containing product |
|---|---:|
| 85123A | 2,104 |
| 85099B | 1,935 |
| 22423 | 1,685 |
| 47566 | 1,593 |
| 20725 | 1,392 |
| 84879 | 1,371 |
| 22197 | 1,296 |
| 22720 | 1,241 |
| 22457 | 1,219 |
| 20727 | 1,216 |

## Research interpretation and limitations

The exported matrix is structurally suitable for binary Apriori analysis. However, the 1,110-item maximum basket warrants a sensitivity study of very large invoices, wholesale orders and special-purpose StockCodes before interpreting association rules as consumer preferences. A binary invoice-product export does **not** include invoice dates, so temporal holdout validation must use the cleaned transactions or original workbook with dates.

The dataset represents one historical UK-based **online retailer**, not a representative sample of UK supermarkets. Association rules do not imply causality.

## Next steps

1. Add a deterministic basket validator that accepts CSV content regardless of a misleading .xls extension and checks IDs, binary encoding, sparsity and basket-size distribution.
2. Analyse unusually large baskets and non-merchandise StockCodes; report exclusions as sensitivity analyses, not undocumented cleaning.
3. Reproduce Apriori support and rule metrics against mlxtend, with threshold sensitivity and unit tests.
4. Re-run chronological holdout evaluation using dated source transactions, without tuning on the holdout.
5. Add verified figures and reproducibility evidence; update README acceptance criteria **only after tests actually pass**.

**Note:** Generated transaction-level exports and raw data remain git-ignored to avoid committing large derived datasets.