import pytest
from projects.project06_fpgrowth_recency.analysis import train_slice,WINDOWS

def test_history_windows_never_see_future():
    for cut in (10,100,1000):
        for window in WINDOWS:
            s=train_slice(cut,window)
            assert 0<= (s.start or 0)<s.stop==cut
            if window=="expanding":
                assert s.start==0
            else:
                assert (s.start or 0)>0

def test_unknown_policy_rejected():
    with pytest.raises(ValueError):
        train_slice(100,"future")
