import numpy as np
def init_weights():
    W = np.random.randn(3, 1)
    b = np.random.randn()
    return 'W and b are inited'

def test_w_bad():
    assert init_weights(123) == 'W and b are inited'

def test_w_bad1():
    assert init_weights('123') == 'W and b are inited'

def test_w_good():
    assert init_weights() == 'W and b are inited'

def test_w_good2():
    assert init_weights() == 'W and b are inited'