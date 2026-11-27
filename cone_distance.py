import numpy as np

from damage_index import damage_index

LOG_TAU_RANGE = 25.0
GRID_SIZE = 200
REFINEMENT_ITERATIONS = 60
GOLDEN = (np.sqrt(5) - 1) / 2


def phi(log_tau, s, eps, c, r2):
    s_d0_squared = s[-1] ** 2
    tau = s_d0_squared * np.exp(log_tau)
    lam = tau - s_d0_squared
    shifted = (s[None, :, None] ** 2 - s_d0_squared) + tau[:, None, :]
    a = np.sum(c[None] ** 2 / shifted ** 2, 1)
    b = np.sum(s[None, :, None] ** 2 * c[None] ** 2 / shifted ** 2, 1)
    return np.sqrt(r2[None] + lam ** 2 * a) - eps * np.sqrt(b)


def golden_section(f, lo, hi, n_iter):
    for _ in range(n_iter):
        a = hi - GOLDEN * (hi - lo)
        b = lo + GOLDEN * (hi - lo)
        left = f(a) < f(b)
        hi = np.where(left, b, hi)
        lo = np.where(left, lo, a)
    return 0.5 * (lo + hi)


def minimize_phi(s, eps, c, r2):
    grid = np.linspace(-LOG_TAU_RANGE, LOG_TAU_RANGE, GRID_SIZE)
    values = phi(grid[:, None] * np.ones((1, c.shape[1])), s, eps, c, r2)
    i = np.argmin(values, 0)
    lo = grid[np.maximum(i - 1, 0)]
    hi = grid[np.minimum(i + 1, GRID_SIZE - 1)]
    best = golden_section(lambda x: phi(x[None], s, eps, c, r2)[0], lo, hi,
                         REFINEMENT_ITERATIONS)
    return phi(best[None], s, eps, c, r2)[0]


def anisotropic_distance(model, c, r2):
    s, eps = model["St1"], model["epsbar"]
    distance = np.zeros(c.shape[1])
    outside = damage_index(model, c, r2) > 1
    if outside.any():
        norm_w_bar = np.sqrt(np.sum(c[:, outside] ** 2, 0) + r2[outside])
        inf_phi = minimize_phi(s, eps, c[:, outside], r2[outside])
        distance[outside] = np.maximum(np.minimum(inf_phi, norm_w_bar), 0)
    return distance, outside
