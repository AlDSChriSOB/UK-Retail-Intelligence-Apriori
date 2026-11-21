import numpy as np
from scipy.sparse import csr_matrix
from projects.project02_fpgrowth.recommendation import pair_rules_from_itemsets
from src.robust_benchmark import rank_items

def test_matched_itemset_rules_and_predictions():
    itemsets={("a",):.8,("b",):.8,("c",):.4,("a","b"):.6,("a","c"):.4}
    rules=pair_rules_from_itemsets(itemsets,["a","b","c"],5,.2)
    assert abs(rules[0,1]-.75)<1e-12
    assert abs(rules[1,0]-.75)<1e-12
    assert abs(rules[0,2]-.5)<1e-12
    freq=np.array([4,4,2])
    co=csr_matrix(np.array([[0,3,2],[3,0,0],[2,0,0]]))
    assert rank_items({0},freq,co,rules,"apriori",2)==[1,2]

def test_confidence_filter():
    itemsets={("a",):.8,("b",):.8,("a","b"):.1}
    rules=pair_rules_from_itemsets(itemsets,["a","b"],5,.2)
    assert rules.nnz==0
