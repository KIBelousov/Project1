from LogReg import LogisticRegression
import numpy as np
def test_cost_function():
    a = LogisticRegression()
    assert a.cost_function(np.array([0.5,0.5]),np.array([0.5,0.5])) == 0.6931471805599453

def test_cost_function_1():
    a = LogisticRegression()
    assert a.cost_function(np.array([0.33,0.81]),np.array([0.11,0.5])) == 0.7070520210187353