from LogReg import LogisticRegression
def test_sigm():
    a = LogisticRegression()
    assert a.sigm(2) == 0.8807970779778823