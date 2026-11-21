import numpy as np
from scipy.sparse import csr_matrix
from projects.project02_fpgrowth.benchmark import sparse_binary_frame,run_miner,canonical_itemsets
from projects.project02_fpgrowth.recommendation import pair_rules_from_itemsets

def test_threshold_monotonicity_and_rule_complexity():
    x=csr_matrix(np.array([[1,1,0],[1,1,1],[1,0,1],[0,1,1],[1,1,0]],dtype=bool))
    codes=["a","b","c"]
    frame=sparse_binary_frame(x,codes)
    low=canonical_itemsets(run_miner(frame,"fpgrowth",.2,2)[0])
    high=canonical_itemsets(run_miner(frame,"fpgrowth",.6,2)[0])
    assert set(high).issubset(low)
    for k in high:
        assert abs(high[k]-low[k])<1e-12
    low_rules=pair_rules_from_itemsets(low,codes,5,.2)
    high_rules=pair_rules_from_itemsets(low,codes,5,.8)
    assert high_rules.nnz<=low_rules.nnz
