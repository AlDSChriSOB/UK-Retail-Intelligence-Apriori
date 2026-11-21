import numpy as np
import pytest
from projects.project07_fpgrowth_cold_start.analysis import frequency_strata

def test_frequency_strata_train_only_and_unseen():
    labels,q=frequency_strata(np.array([0,1,2,3,4,5,6]))
    assert labels[0]=="unseen"
    assert labels[1]=="rare"
    assert labels[-1]=="popular"
    assert q["tertile_1"]<=q["tertile_2"]

def test_empty_train_rejected():
    with pytest.raises(ValueError):
        frequency_strata(np.array([0,0,0]))
