"""Validate a binary invoice-by-product basket CSV, regardless of filename extension.

Usage: python -m src.validate_basket "path/to/uk_invoice_product_basket.xls"
The input is expected to be CSV text, NOT a binary Excel workbook.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def validate_basket(path: str | Path) -> dict:
    """Stream CSV basket rows; raise ValueError for structural/data errors."""
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        prefix = handle.read(8)
        handle.seek(0)
        if prefix.startswith("PK\x03\x04") or prefix.startswith("\xd0\xcf\x11\xe0"):
            raise ValueError("Binary Excel workbook detected; export it as CSV first.")
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError("Expected invoice ID column and at least one product column.")
        product_codes = [h.strip() for h in header[1:]]
        if not header[0].strip():
            raise ValueError("Invoice identifier header is blank.")
        if any(not code for code in product_codes):
            raise ValueError("Blank product code header.")
        if len(product_codes) != len(set(product_codes)):
            raise ValueError("Duplicate product code headers.")
        invoices = set()
        product_counts = [0] * len(product_codes)
        size_counts = Counter()
        rows = 0
        nonzero = 0
        for line, row in enumerate(reader, start=2):
            if len(row) != len(header):
                raise ValueError(f"CSV line {line}: expected {len(header)} columns, got {len(row)}.")
            invoice = row[0].strip()
            if not invoice or invoice in invoices:
                raise ValueError(f"CSV line {line}: blank or duplicate invoice ID {invoice!r}.")
            invoices.add(invoice)
            size = 0
            for j, value in enumerate(row[1:]):
                v = value.strip().lower()
                if v in ("1", "true"):
                    product_counts[j] += 1
                    size += 1
                elif v not in ("0", "false"):
                    raise ValueError(f"CSV line {line}, product {product_codes[j]}: not binary: {value!r}.")
            if size == 0:
                raise ValueError(f"CSV line {line}: empty basket.")
            size_counts[size] += 1
            nonzero += size
            rows += 1
    if not rows:
        raise ValueError("Basket contains no invoice rows.")
    n_products = len(product_codes)
    top = sorted(zip(product_codes, product_counts), key=lambda x: (-x[1], x[0]))[:10]
    return {
        "input_file": str(path),
        "invoice_count": rows,
        "product_count": n_products,
        "product_presence_count": nonzero,
        "average_basket_size": nonzero / rows,
        "maximum_basket_size": max(size_counts),
        "density": nonzero / (rows * n_products),
        "sparsity": 1 - nonzero / (rows * n_products),
        "basket_size_histogram": dict(sorted(size_counts.items())),
        "top_product_codes": [
            {"stock_code": code, "invoice_count": count} for code, count in top
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Path to CSV-formatted basket (any extension)")
    parser.add_argument("--output", default="reports/generated/basket_validation.json")
    args = parser.parse_args()
    result = validate_basket(args.path)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
