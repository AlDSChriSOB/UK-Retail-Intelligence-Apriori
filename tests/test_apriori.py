import math
from src.apriori import apriori, generate_rules


def test_frequent_itemsets_and_support():
    baskets = [
        ["A", "B", "C"],
        ["A", "B"],
        ["A", "C"],
        ["B", "C"],
    ]
    result = apriori(baskets, min_support=0.5, max_len=3)
    supports = {frozenset(r.itemsets): r.support for r in result.itertuples()}
    assert math.isclose(supports[frozenset(["A"])], 0.75)
    assert math.isclose(supports[frozenset(["A", "B"])], 0.50)
    assert frozenset(["A", "B", "C"]) not in supports


def test_rule_metrics():
    baskets = [["A", "B"], ["A", "B"], ["A"], ["B"]]
    itemsets = apriori(baskets, min_support=0.25, max_len=2)
    rules = generate_rules(itemsets, min_confidence=0.5)
    rule = rules[
        rules["antecedents"].map(lambda x: x == frozenset(["A"])) &
        rules["consequents"].map(lambda x: x == frozenset(["B"]))
    ].iloc[0]
    assert math.isclose(rule["support"], 0.5)
    assert math.isclose(rule["confidence"], 2/3)
    assert math.isclose(rule["lift"], 8/9)


def test_empty_input():
    assert apriori([], min_support=0.5).empty


def test_invalid_support():
    try:
        apriori([["A"]], min_support=0)
    except ValueError:
        return
    raise AssertionError("Expected ValueError")
