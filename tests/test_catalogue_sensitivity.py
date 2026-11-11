import pandas as pd
from src.catalogue_sensitivity import candidate_mapping


def test_identical_primary_descriptions_merge_but_variants_do_not():
    df = pd.DataFrame({
        "StockCode": ["A", "A", "B", "C", "D", "D"],
        "Description": [" Red Mug ", "RED MUG", "red  mug", "BLUE MUG", "PLATE", "plate"],
    })
    mapping, groups = candidate_mapping(df)
    assert mapping == {"A": "A", "B": "A"}
    assert len(groups) == 1
    assert "C" not in mapping
    assert "D" not in mapping


def test_primary_description_uses_most_frequent_value():
    df = pd.DataFrame({
        "StockCode": ["A", "A", "A", "B", "C"],
        "Description": ["MUG", "MUG", "CUP", "MUG", "CUP"],
    })
    mapping, groups = candidate_mapping(df)
    assert mapping["B"] == "A"
    assert "C" not in mapping
