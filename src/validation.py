"""Reference validation helpers."""
from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori as mlxtend_apriori


def compare_itemsets(
    basket: pd.DataFrame,
    min_support: float = 0.01,
    max_len: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare the independent Apriori implementation with mlxtend."""
    reference = mlxtend_apriori(
        basket.astype(bool),
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
        low_memory=True,
    )
    reference["itemsets"] = reference["itemsets"].map(frozenset)

    from .apriori import apriori
    own = apriori(
        (row.index[row.to_numpy(dtype=bool)].tolist()
         for _, row in basket.iterrows()),
        min_support=min_support,
        max_len=max_len,
    )
    left = reference.rename(columns={"support": "support_reference"})
    right = own.rename(columns={"support": "support_own"})
    comparison = left.merge(right, on="itemsets", how="outer", indicator=True)
    comparison["support_abs_difference"] = (
        comparison["support_reference"] - comparison["support_own"]
    ).abs()
    return reference, own, comparison
