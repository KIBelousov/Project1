from LogReg import LogisticRegression
def test_accuracy():
    a = LogisticRegression()
    assert a.accuracy([1,2],[1,1]) == 0.5

def test_accuracy_2():
    a = LogisticRegression()
    assert a.accuracy([1,2],[3,1]) == 1.0