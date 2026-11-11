"""Deterministic tests for the streamed basket validator."""
import csv

import pytest

from src.validate_basket import validate_basket


def write_basket(tmp_path, rows, name="basket.xls"):
    path = tmp_path / name
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)
    return path


def test_csv_with_xls_extension_is_valid(tmp_path):
    path = write_basket(tmp_path, [
        ["InvoiceNo", "A", "B", "C"],
        ["100", "1", "0", "true"],
        ["101", "0", "1", "0"],
        ["102", "false", "1", "1"],
    ])
    report = validate_basket(path)
    assert report["invoice_count"] == 3
    assert report["product_count"] == 3
    assert report["product_presence_count"] == 5
    assert report["maximum_basket_size"] == 2
    assert report["density"] == pytest.approx(5 / 9)
    assert report["top_product_codes"][0]["invoice_count"] == 2


@pytest.mark.parametrize("rows,expected", [
    ([["InvoiceNo", "A", "A"], ["1", "1", "0"]], "Duplicate product"),
    ([["InvoiceNo", "A"], ["1", "1"], ["1", "0"]], "duplicate invoice"),
    ([["InvoiceNo", "A"], ["1", "2"]], "not binary"),
    ([["InvoiceNo", "A"], ["1", "0"]], "empty basket"),
    ([["InvoiceNo", "A", "B"], ["1", "1"]], "expected 3 columns"),
])
def test_rejects_invalid_baskets(tmp_path, rows, expected):
    with pytest.raises(ValueError, match=expected):
        validate_basket(write_basket(tmp_path, rows))


def test_rejects_binary_xlsx_signature(tmp_path):
    path = tmp_path / "basket.xls"
    path.write_bytes(b"PK\\x03\\x04bad")
    with pytest.raises((ValueError, UnicodeDecodeError)):
        validate_basket(path)
