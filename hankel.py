import numpy as np


def hankel(z, L):
    return np.column_stack([z[j:j + L].ravel() for j in range(len(z) - L + 1)])
