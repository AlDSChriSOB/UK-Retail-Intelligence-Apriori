import numpy as np
import pytest
from scipy.sparse import csr_matrix
from projects.project09_eclat.analysis import eclat

def test_eclat_supports_and_intersections():
    mat=csr_matrix(np.array([[1,1,0],[1,0,1],[1,1,1],[0,1,1]],dtype=np.int32))
    out=eclat(mat,["A","B","C"],.5,3)
    assert out[("A",)]==.75
    assert out[("A","B")]==.5
    assert out[("B","C")]==.5
    assert ("A","B","C") not in out

def test_minimum_count_uses_ceil():
    mat=csr_matrix(np.array([[1,0],[0,1],[0,1]],dtype=np.int32))
    assert ("A",) not in eclat(mat,["A","B"],.34,2)
    assert eclat(mat,["A","B"],.34,2)[("B",)]==2/3

def test_invalid_parameters():
    with pytest.raises(ValueError):eclat(csr_matrix([[1]]),["A"],0,2)
    with pytest.raises(ValueError):eclat(csr_matrix([[1]]),["A"],.5,0)
