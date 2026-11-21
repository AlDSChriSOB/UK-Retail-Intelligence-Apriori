import numpy as np
import pytest
from scipy.sparse import csr_matrix
from projects.project03_eclat.mining import eclat_itemsets
from projects.project02_fpgrowth.benchmark import sparse_binary_frame,run_miner,canonical_itemsets
from projects.project02_fpgrowth.recommendation import pair_rules_from_itemsets

def test_eclat_matches_both_miners():
    x=csr_matrix(np.array([[1,1,0],[1,1,1],[1,0,1],[0,1,1],[1,1,0]],dtype=bool))
    codes=["a","b","c"]
    e=eclat_itemsets(x,codes,.4,2)
    table=sparse_binary_frame(x,codes)
    for algorithm in ("apriori","fpgrowth"):
        mined=canonical_itemsets(run_miner(table,algorithm,.4,2)[0])
        assert e.keys()==mined.keys()
        assert all(abs(e[k]-mined[k])<1e-12 for k in e)
    assert abs(e[("a","b")]-.6)<1e-12
    assert pair_rules_from_itemsets(e,codes,5,.2)[0,1]>0

def test_eclat_depth_three():
    x=csr_matrix(np.ones((4,3),dtype=bool))
    result=eclat_itemsets(x,["a","b","c"],.5,3)
    assert ("a","b","c") in result
    assert len(result)==7

def test_eclat_bad_support():
    with pytest.raises(ValueError):
        eclat_itemsets(csr_matrix(np.eye(2)),["a","b"],0)
