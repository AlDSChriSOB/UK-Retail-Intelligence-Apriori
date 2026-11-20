import numpy as np
from scipy.sparse import csr_matrix
from src.robust_benchmark import rank_items
from src.recommendation_benchmark import fit_models

def test_all_methods_exclude_seen_and_unseen():
    mat=csr_matrix(np.array([[1,1,0,0],[1,0,1,0],[0,1,1,0]],dtype=np.int64))
    freq,co,rules=fit_models(mat,min_support=.2,min_confidence=.1)
    for method in ("apriori","cooccurrence","cosine","popularity"):
        ranking=rank_items({0},freq,co,rules,method,10)
        assert 0 not in ranking
        assert 3 not in ranking
        assert len(ranking)==2

def test_cosine_prefers_associated_product():
    mat=csr_matrix(np.array([[1,1,0],[1,1,0],[0,0,1]],dtype=np.int64))
    freq,co,rules=fit_models(mat,.2,.1)
    assert rank_items({0},freq,co,rules,"cosine",2)[0]==1
