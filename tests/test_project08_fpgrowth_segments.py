import pandas as pd
import pytest
from projects.project08_fpgrowth_segments.analysis import rfm_training_only

def test_rfm_uses_training_invoices_only():
    tx=pd.DataFrame([
        {"InvoiceNo":"1","InvoiceDate":pd.Timestamp("2020-01-01"),"CustomerID":1.0,"Quantity":1,"UnitPrice":10.0},
        {"InvoiceNo":"2","InvoiceDate":pd.Timestamp("2020-01-02"),"CustomerID":2.0,"Quantity":2,"UnitPrice":10.0},
        {"InvoiceNo":"3","InvoiceDate":pd.Timestamp("2020-01-03"),"CustomerID":1.0,"Quantity":100,"UnitPrice":100.0}])
    grouped=pd.DataFrame({"InvoiceNo":["1","2","3"],"date":pd.to_datetime(["2020-01-01","2020-01-02","2020-01-03"])})
    first,info=rfm_training_only(tx,grouped,2)
    changed=tx.copy()
    changed.loc[2,"Quantity"]=999999
    second,_=rfm_training_only(changed,grouped,2)
    assert first==second
    assert info["training_customers"]==2
    assert all(x in ("low","middle","high") for x in first.values())

def test_empty_known_customers_rejected():
    tx=pd.DataFrame([{"InvoiceNo":"1","InvoiceDate":pd.Timestamp("2020-01-01"),"CustomerID":None,"Quantity":1,"UnitPrice":1.}])
    grouped=pd.DataFrame({"InvoiceNo":["1"],"date":[pd.Timestamp("2020-01-01")]})
    with pytest.raises(ValueError):rfm_training_only(tx,grouped,1)
