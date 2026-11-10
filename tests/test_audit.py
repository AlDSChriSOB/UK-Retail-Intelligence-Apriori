import pandas as pd

from src.audit import audit, clean
from src.data import build_basket_matrix


def sample_data():
    return pd.DataFrame({
        "InvoiceNo": ["100", "100", "C101", "102", "103", "104"],
        "StockCode": ["A", "A", "B", "C", "D", "E"],
        "Description": ["Item A", "Item A", "Item B", "Item C", "Item D", None],
        "Quantity": [2, 1, -1, 0, 3, 1],
        "InvoiceDate": pd.to_datetime([
            "2011-01-01", "2011-01-01", "2011-01-02",
            "2011-01-03", "2011-01-04", "2011-01-05"
        ]),
        "UnitPrice": [1.0, 1.0, 1.0, 1.0, 0.0, 2.0],
        "CustomerID": [None, None, 2, 3, 4, 5],
        "Country": ["United Kingdom", "United Kingdom", "United Kingdom",
                    "United Kingdom", "United Kingdom", "France"],
    })


def test_audit_profiles_duplicates_cancellations_and_anomalies():
    df = sample_data()
    overview, missing, numeric, countries, monthly = audit(df)
    assert overview["rows"] == 6
    assert overview["full_row_duplicates"] == 0
    assert overview["cancellation_lines"] == 1
    assert overview["negative_quantity_lines"] == 1
    assert overview["zero_quantity_lines"] == 1
    assert overview["zero_unit_price_lines"] == 1
    assert overview["blank_descriptions"] == 1
    assert set(missing.columns) == {"column", "missing_count", "missing_pct"}
    assert len(numeric) == 2
    assert not countries.empty and not monthly.empty


def test_cleaning_is_sequential_and_preserves_anonymous_uk_baskets():
    df = sample_data()
    uk, all_country, stages = clean(df)
    assert len(all_country) == 3
    assert len(uk) == 2
    assert uk.CustomerID.isna().all()
    assert uk.InvoiceNo.nunique() == 1
    basket = build_basket_matrix(uk)
    assert basket.shape == (1, 1)
    assert basket.iloc[0, 0] == 1
    assert stages.iloc[-1]["stage"] == "UK_primary_population"
