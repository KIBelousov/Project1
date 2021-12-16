import numpy as np
def sigm(x):
    return 1 / (1 + np.exp(-x))

def test_sigm_good_int():
    assert sigm(2) == 0.8807970779778823

def test_sigm_good_int_2():
    assert sigm(1256) == 1.0

def test_sigm_bad_str():
    assert sigm('a') == 0.8807970779778823

def test_sigm_bad_int():
    assert sigm(-2) == 0.8807970779778823