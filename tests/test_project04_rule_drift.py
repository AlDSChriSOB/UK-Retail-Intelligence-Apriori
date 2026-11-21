import numpy as np
import pytest
from scipy.sparse import csr_matrix
from projects.project04_rule_drift.analysis import confidence_for_rules,paired_bootstrap_confidence_drift

def test_out_of_time_conditional_confidence():
    matrix=csr_matrix([[1,1,0],[1,0,1],[0,1,1]])
    conf=confidence_for_rules(matrix,[(0,1),(0,2),(2,0)])
    assert np.allclose(conf,[.5,.5,.5])

def test_undefined_antecedent_excluded():
    matrix=csr_matrix([[1,0,0],[1,0,0]])
    conf=confidence_for_rules(matrix,[(1,0),(0,1)])
    assert np.isnan(conf[0]) and conf[1]==0
    summary=paired_bootstrap_confidence_drift([.3,.5],[np.nan,.1],100,12)
    assert summary["n_rules"]==1
    assert abs(summary["mean_change"]+.4)<1e-12

def test_bad_bootstrap():
    with pytest.raises(ValueError):
        paired_bootstrap_confidence_drift([.1],[.2],0)
