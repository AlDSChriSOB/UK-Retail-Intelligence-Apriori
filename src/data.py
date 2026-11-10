"""Dataset acquisition, audit, cleaning and basket construction."""
from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import requests
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
ZIP_PATH = Path("data/raw/online-retail.zip")
RAW_DIR = Path("data/raw")
EXCEL_PATH = RAW_DIR / "Online Retail.xlsx"


def download_dataset(url: str = DATA_URL, zip_path: Path = ZIP_PATH) -> Path:
    """Download the official UCI archive if it is not already present."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        zip_path.write_bytes(response.content)
    return zip_path


def extract_dataset(zip_path: Path = ZIP_PATH, extract_dir: Path = RAW_DIR) -> Path:
    """Extract the UCI archive and return the workbook path."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    with ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    candidates = list(extract_dir.rglob("*.xlsx"))
    if not candidates:
        raise FileNotFoundError("No .xlsx file found in the downloaded UCI archive.")
    return candidates[0]


def load_transactions(path: Path = EXCEL_PATH) -> pd.DataFrame:
    """Load the workbook and validate its schema."""
    df = pd.read_excel(path, engine="openpyxl")
    df.columns = [str(c).strip() for c in df.columns]
    required = {
        "InvoiceNo", "StockCode", "Description", "Quantity",
        "InvoiceDate", "UnitPrice", "CustomerID", "Country"
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Unexpected dataset schema; missing columns: {sorted(missing)}")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    return df


def audit_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Return a compact data-quality audit table."""
    rows = [
        ("rows", len(df)),
        ("columns", len(df.columns)),
        ("missing_invoice", int(df["InvoiceNo"].isna().sum())),
        ("missing_stock_code", int(df["StockCode"].isna().sum())),
        ("missing_description", int(df["Description"].isna().sum())),
        ("missing_customer_id", int(df["CustomerID"].isna().sum())),
        ("missing_invoice_date", int(df["InvoiceDate"].isna().sum())),
        ("cancellation_rows", int(df["InvoiceNo"].astype("string").str.startswith("C", na=False).sum())),
        ("non_positive_quantity_rows", int((df["Quantity"] <= 0).sum())),
        ("non_positive_unit_price_rows", int((df["UnitPrice"] <= 0).sum())),
        ("duplicate_full_rows", int(df.duplicated().sum())),
        ("unique_invoices", int(df["InvoiceNo"].nunique())),
        ("unique_stock_codes", int(df["StockCode"].nunique())),
        ("unique_countries", int(df["Country"].nunique())),
    ]
    return pd.DataFrame(rows, columns=["audit_measure", "value"])


def clean_transactions(
    df: pd.DataFrame,
    country: str = "United Kingdom",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the documented policy; retain missing CustomerID for invoice baskets."""
    counts: dict[str, int] = {"input_rows": len(df)}
    x = df.copy()
    x = x.dropna(subset=["InvoiceNo", "StockCode", "InvoiceDate"])
    counts["after_required_fields"] = len(x)

    is_cancel = x["InvoiceNo"].astype("string").str.startswith("C", na=False)
    counts["cancellation_rows_removed"] = int(is_cancel.sum())
    x = x.loc[~is_cancel].copy()

    bad_qty = x["Quantity"] <= 0
    counts["non_positive_quantity_rows_removed"] = int(bad_qty.sum())
    x = x.loc[~bad_qty].copy()

    bad_price = x["UnitPrice"] <= 0
    counts["non_positive_unit_price_rows_removed"] = int(bad_price.sum())
    x = x.loc[~bad_price].copy()

    country_mask = x["Country"].astype("string").str.strip().str.casefold() == country.casefold()
    counts["non_target_country_rows_removed"] = int((~country_mask).sum())
    x = x.loc[country_mask].copy()

    x["StockCode"] = x["StockCode"].astype(str).str.strip()
    x["InvoiceNo"] = x["InvoiceNo"].astype(str).str.strip()
    x = x.loc[x["StockCode"].ne("") & x["InvoiceNo"].ne("")].copy()
    counts["final_rows"] = len(x)
    counts["final_invoices"] = int(x["InvoiceNo"].nunique())
    counts["final_products"] = int(x["StockCode"].nunique())
    return x, counts


def build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Create an invoice × StockCode binary incidence matrix."""
    if df.empty:
        raise ValueError("No transaction rows available after cleaning.")
    return (
        df.assign(present=1)
          .pivot_table(index="InvoiceNo", columns="StockCode", values="present",
                       aggfunc="max", fill_value=0)
          .astype("uint8")
          .rename_axis(index="InvoiceNo", columns="StockCode")
    )


if __name__ == "__main__":
    archive = download_dataset()
    workbook = extract_dataset(archive)
    data = load_transactions(workbook)
    print(audit_transactions(data).to_string(index=False))
    cleaned, summary = clean_transactions(data)
    print("\nCleaning summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    basket = build_basket_matrix(cleaned)
    print(f"\nBasket matrix shape: {basket.shape}")
