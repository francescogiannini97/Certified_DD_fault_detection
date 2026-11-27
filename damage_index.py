import numpy as np


def damage_index(model, c, r2):
    s, eps = model["St1"], model["epsbar"]
    denominator = eps ** 2 * np.sum(c ** 2 / (s[:, None] ** 2 - eps ** 2), 0)
    D = np.zeros_like(r2)
    positive = denominator > 0
    D[positive] = r2[positive] / denominator[positive]
    D[~positive & (r2 > 0)] = np.inf
    return D
