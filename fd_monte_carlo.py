import os

import numpy as np

from cone_distance import anisotropic_distance
from hankel import hankel
from isotropic_test import isotropic_distance
from output import RESULTS
from records import record, rms, uniform_noise
from shear_frame import shear_frame
from weighting import (ball_W0, coordinates, ellipsoidal_W0, falsified, no_certificate, weight,
                       weighted_svd, weighted_windows)

SEED = 2026
m, n, L, N = 1, 6, 10, 4000
d0 = m * L + n
TEST_LENGTH = 640
GAMMAS = [1e-3, 3e-3, 1e-2]
DELTAS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
N_RUNS = 10
TESTS = ["isotropic, ball (gap-based)", "anisotropic, ball",
         "isotropic, ellipsoidal", "anisotropic, ellipsoidal"]
delta = np.sqrt(L)


def normalized_distances(model, w_noisy):
    if no_certificate(model) or falsified(model):
        nan = np.full(w_noisy.shape[0] - L + 1, np.nan)
        return nan, nan, None
    c, r2 = coordinates(model, weighted_windows(model, w_noisy))
    anisotropic, outside = anisotropic_distance(model, c, r2)
    outside_cone = dict(St1=model["St1"], epsbar=model["epsbar"], c=c[:, outside],
                        r2=r2[outside], distance=anisotropic[outside])
    return isotropic_distance(model, c, r2) / delta, anisotropic / delta, outside_cone


def run():
    rng = np.random.default_rng(SEED)
    systems = {D: shear_frame(D) for D in DELTAS}
    distances = np.zeros((len(GAMMAS), len(DELTAS), N_RUNS, TEST_LENGTH - L + 1, 4))
    info = {g: [] for g in GAMMAS}
    outside_cones = []
    for a, gamma in enumerate(GAMMAS):
        for run_index in range(N_RUNS):
            wd = record(rng, systems[0.0], N)
            eta = gamma * rms(wd)
            ed = uniform_noise(rng, N, eta)
            ball = weighted_svd(wd + ed, ball_W0(eta), L, d0)
            ellipsoidal = weighted_svd(wd + ed, ellipsoidal_W0(eta), L, d0)
            info[gamma].append(dict(
                fbar_ball=ball["epsbar"] / ball["St1"][-1],
                fbar_ellipsoidal=ellipsoidal["epsbar"] / ellipsoidal["St1"][-1],
                cond_ball=ball["St1"][0] / ball["St1"][-1],
                cond_ellipsoidal=ellipsoidal["St1"][0] / ellipsoidal["St1"][-1],
                WE_ratio=np.linalg.norm(hankel(weight(ed, ellipsoidal["W0"]), L), 2)
                / ellipsoidal["epsbar"],
                rank=np.linalg.matrix_rank(hankel(wd, L)),
                stops=sum(bool(no_certificate(x) or falsified(x)) for x in (ball, ellipsoidal)),
                falsified=sum(bool(falsified(x)) for x in (ball, ellipsoidal))))
            for b, D in enumerate(DELTAS):
                wt = record(rng, systems[D], TEST_LENGTH)
                w_noisy = wt + uniform_noise(rng, TEST_LENGTH, eta)
                for k, model in enumerate((ball, ellipsoidal)):
                    iso, an, outside = normalized_distances(model, w_noisy)
                    distances[a, b, run_index, :, 2 * k] = iso
                    distances[a, b, run_index, :, 2 * k + 1] = an
                    if outside is not None:
                        outside_cones.append(outside)
        print(f"fd Monte Carlo: gamma = {100 * gamma:g}% done", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    np.savez_compressed(os.path.join(RESULTS, "fd_distances.npz"), distances=distances,
                        gammas=GAMMAS, deltas=DELTAS,
                        **{key: np.array([[x[key] for x in info[g]] for g in GAMMAS])
                           for key in info[GAMMAS[0]][0]})
    return dict(distances=distances, info=info, outside_cones=outside_cones)


def load_distances():
    return np.load(os.path.join(RESULTS, "fd_distances.npz"))["distances"]
