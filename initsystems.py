import numpy as np


def init_random_system(n):
    x = np.random.randn(n, 3)
    p = 2 * np.random.rand(n, 3) - 1
    q = 2 * np.random.rand(n, 3) - 1

    return x, p, q

