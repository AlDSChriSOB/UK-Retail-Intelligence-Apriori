import numpy as np
from scipy.sparse import csr_matrix
from projects.project02_fpgrowth.benchmark import sparse_binary_frame, run_miner, canonical_itemsets

def test_exact_frequent_itemsets():
    x=csr_matrix(np.array([[1,1,0],[1,1,1],[1,0,1],[0,1,1],[1,1,0]],dtype=bool))
    table=sparse_binary_frame(x,["a","b","c"])
    a,_,_=run_miner(table,"apriori",.4)
    f,_,_=run_miner(table,"fpgrowth",.4)
    aa,ff=canonical_itemsets(a),canonical_itemsets(f)
    assert aa.keys()==ff.keys()
    assert all(abs(aa[k]-ff[k])<1e-12 for k in aa)
    assert ("a","b") in aa

def test_max_len_two():
    x=csr_matrix(np.ones((5,3),dtype=bool))
    table=sparse_binary_frame(x,["a","b","c"])
    for method in ("apriori","fpgrowth"):
        mined,_,_=run_miner(table,method,.5,max_len=2)
        assert max(len(i) for i in mined["itemsets"])==2
