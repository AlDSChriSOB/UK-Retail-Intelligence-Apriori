import pytest
from src.full_apriori import apriori, association_rules, generate_candidates

BASKETS = [
    {"A", "B", "C"}, {"A", "B", "C"}, {"A", "B"},
    {"A", "C"}, {"B", "C"},
]

def test_candidate_pruning():
    assert generate_candidates([("A", "B"), ("A", "C"), ("B", "C")], 3) == [("A", "B", "C")]
    assert generate_candidates([("A", "B"), ("A", "C")], 3) == []

def test_three_itemset_and_support():
    found = apriori(BASKETS, min_support=0.4)
    assert found[frozenset({"A", "B", "C"})] == pytest.approx(0.4)
    assert found[frozenset({"A", "B"})] == pytest.approx(0.6)
    assert len(found) == 7

def test_rules_metrics():
    rules = association_rules(apriori(BASKETS, 0.4), 0.6)
    rule = next(r for r in rules if r["antecedents"] == frozenset({"A", "B"}) and r["consequents"] == frozenset({"C"}))
    assert rule["confidence"] == pytest.approx(2 / 3)
    assert rule["lift"] == pytest.approx((2 / 3) / (4 / 5))
    assert rule["support"] == pytest.approx(0.4)

def test_duplicate_items_do_not_inflate_support():
    assert apriori([["A", "A"], ["B"]], 0.5)[frozenset({"A"})] == 0.5

def test_validation():
    with pytest.raises(ValueError):
        apriori([], 0.5)
    with pytest.raises(ValueError):
        apriori(BASKETS, 0)
    with pytest.raises(ValueError):
        apriori(BASKETS, 0.5, max_len=0)

def test_mlxtend_equivalence():
    pd = pytest.importorskip("pandas")
    ml = pytest.importorskip("mlxtend.frequent_patterns")
    frame = pd.DataFrame([{item: item in basket for item in ("A", "B", "C")} for basket in BASKETS])
    expected = ml.apriori(frame, min_support=0.4, use_colnames=True)
    ours = apriori(BASKETS, 0.4)
    assert set(expected["itemsets"]) == set(ours)
    for row in expected.itertuples():
        assert ours[row.itemsets] == pytest.approx(row.support)
    reference = ml.association_rules(expected, metric="confidence", min_threshold=0.6, num_itemsets=len(frame))
    actual = {(r["antecedents"], r["consequents"]): r for r in association_rules(ours, 0.6)}
    assert len(actual) == len(reference)
    for row in reference.itertuples(index=False):
        key = (row.antecedents, row.consequents)
        assert actual[key]["confidence"] == pytest.approx(row.confidence)
        assert actual[key]["lift"] == pytest.approx(row.lift)
