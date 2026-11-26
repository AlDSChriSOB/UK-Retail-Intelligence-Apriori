from src.project10 import mine
import pytest

@pytest.mark.parametrize('name',['eclat','apriori','fpgrowth'])
def test_exact(name):
    baskets=[['a','b','c'],['a','b'],['a','c'],['b']]
    assert mine(name,baskets,2,3)=={('a',):3,('b',):3,('c',):2,('a','b'):2,('a','c'):2}
