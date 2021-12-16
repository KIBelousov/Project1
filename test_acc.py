from sklearn.metrics import accuracy_score
import numpy as np

def accuracy(y, p):
    return accuracy_score(p, y)

def test_acc_bad():
    assert accuracy_score(1, 'a') == 0.1

def test_acc_bad1():
    assert accuracy_score(np.array([[0, 1], [1, 1, 1]]), np.ones((2, 2))) == 1 

def test_acc_good():
    assert accuracy_score(np.array([[0, 1], [1, 1]]) ,np.ones((2, 2))) == 0.5

def test_acc_good1():
    assert accuracy_score(np.array([[0, 0], [0, 0]]), np.zeros((2, 2))) == 1.0   