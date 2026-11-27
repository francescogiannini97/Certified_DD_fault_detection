import numpy as np


def isotropic_distance(model, c, r2):
    fbar = model["epsbar"] / model["St1"][-1]
    norm_c = np.sqrt(np.sum(c ** 2, 0))
    norm_r = np.sqrt(r2)
    norm_w_bar = np.sqrt(norm_c ** 2 + r2)
    alpha = np.arctan2(norm_r, norm_c)
    half_aperture = np.arcsin(fbar)
    return np.where(alpha > half_aperture, norm_w_bar * np.sin(alpha - half_aperture), 0.0)
