import os

import numpy as np
import matplotlib.pyplot as plt

from cone_distance import anisotropic_distance
from hankel import hankel
from output import BLUE, GREY, RESULTS, TEXT_WIDTH, save_figure, save_table
from parity_test import parity_statistic, parity_test
from records import q, bounded_drift, record, rms, uniform_noise
from shear_frame import SAMPLING_RATE, shear_frame
from weighting import (coordinates, ellipsoidal_W0, falsified, no_certificate, weighted_svd,
                       weighted_windows)

NOMINAL_SEED, STRESS_SEED = 42, 43
m, n, L, N = 1, 6, 10, 4000
d0 = m * L + n
TEST_LENGTH = 640
N_WINDOWS = TEST_LENGTH - L + 1
GAMMA = 3e-3
ALPHA = 0.01
DELTAS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
NOMINAL_RUNS, STRESS_RUNS = 50, 100
STRESS_LOSS = 0.05
delta = np.sqrt(L)


def setup(rng, healthy):
    wd = record(rng, healthy, N)
    eta = GAMMA * rms(wd)
    wd_noisy = wd + uniform_noise(rng, N, eta)
    wc = record(rng, healthy, N)
    wc_noisy = wc + uniform_noise(rng, N, eta)
    model = weighted_svd(wd_noisy, ellipsoidal_W0(eta), L, d0)
    return dict(eta=eta, model=model, parity=parity_test(model, wc_noisy, ALPHA, N_WINDOWS),
                rank=np.linalg.matrix_rank(hankel(wd, L)),
                stop=bool(no_certificate(model) or falsified(model)))


def decide(run, w_noisy):
    model = run["model"]
    w_bar = weighted_windows(model, w_noisy)
    certified = False
    if not run["stop"]:
        c, r2 = coordinates(model, w_bar)
        certified = bool((anisotropic_distance(model, c, r2)[0] > delta).any())
    statistic = parity_statistic(model, run["parity"], w_bar)
    parity = bool((statistic > run["parity"]["threshold"]).any())
    return certified, parity


def nominal():
    rng = np.random.default_rng(NOMINAL_SEED)
    systems = {D: shear_frame(D) for D in DELTAS}
    decisions = np.zeros((NOMINAL_RUNS, len(DELTAS), 2), bool)
    checks = []
    for k in range(NOMINAL_RUNS):
        run = setup(rng, systems[0.0])
        checks.append((run["rank"], run["stop"]))
        for b, D in enumerate(DELTAS):
            wt = record(rng, systems[D], TEST_LENGTH)
            decisions[k, b] = decide(run, wt + uniform_noise(rng, TEST_LENGTH, run["eta"]))
    return decisions, checks


def stress():
    rng = np.random.default_rng(STRESS_SEED)
    healthy, damaged = shear_frame(0.0), shear_frame(STRESS_LOSS)
    decisions = np.zeros((STRESS_RUNS, 3, 2), bool)
    checks = []
    for k in range(STRESS_RUNS):
        run = setup(rng, healthy)
        checks.append((run["rank"], run["stop"]))
        eta = run["eta"]
        wt = record(rng, healthy, TEST_LENGTH)
        decisions[k, 0] = decide(run, wt + uniform_noise(rng, TEST_LENGTH, eta))
        wt = record(rng, healthy, TEST_LENGTH)
        decisions[k, 1] = decide(run, wt + bounded_drift(rng, TEST_LENGTH, eta, SAMPLING_RATE))
        wt = record(rng, damaged, TEST_LENGTH)
        decisions[k, 2] = decide(run, wt + bounded_drift(rng, TEST_LENGTH, eta, SAMPLING_RATE))
    return decisions, checks


def figure(fractions):
    fig, a = plt.subplots(1, 1, figsize=(0.6 * TEXT_WIDTH, 2.45), layout="constrained")
    x = np.arange(len(DELTAS))
    a.plot(x, fractions[:, 0], ls="-", marker="s", ms=4, color=BLUE, lw=1.2,
           label="certified anisotropic test", mfc=BLUE)
    a.plot(x, fractions[:, 1], ls="--", marker="^", ms=4, color=GREY, lw=1.2,
           label="parity $T^2$ test, $\\alpha=1\\%$", mfc="white")
    a.set_xticks(x)
    a.set_xticklabels([f"{100 * D:g}" for D in DELTAS], fontsize=8.5)
    a.set_xlim(-0.3, len(DELTAS) - 0.7)
    a.set_xlabel("first-story stiffness loss $\\Delta$ [%]")
    a.set_ylim(-0.03, 1.03)
    a.grid(alpha=0.3, lw=0.5)
    a.set_ylabel("records declared faulty")
    fig.legend(loc="outside upper center", ncol=1, frameon=False, handlelength=2.4)
    save_figure(fig, "compare")


def main():
    nominal_decisions, nominal_checks = nominal()
    stress_decisions, stress_checks = stress()
    os.makedirs(RESULTS, exist_ok=True)
    np.savez_compressed(os.path.join(RESULTS, "compare_decisions.npz"),
                        nominal=nominal_decisions, stress=stress_decisions, deltas=DELTAS)
    fractions = nominal_decisions.mean(0)
    save_table("compare", ["Delta [%]", "certified anisotropic", "parity T2"],
               [[f"{100 * D:g}", f"{f[0]:.2f}", f"{f[1]:.2f}"] for D, f in zip(DELTAS, fractions)])
    stress_percent = 100 * stress_decisions.mean(0)
    nominal_loss = 100 * nominal_decisions[:, DELTAS.index(STRESS_LOSS)].mean(0)
    save_table("stress", ["scenario", "runs", "certified (anisotropic) [%]", "parity T2 [%]"], [
        ["healthy, nominal noise", STRESS_RUNS, *[f"{x:g}" for x in stress_percent[0]]],
        ["healthy, bounded drift", STRESS_RUNS, *[f"{x:g}" for x in stress_percent[1]]],
        ["5% loss, nominal noise", NOMINAL_RUNS, *[f"{x:g}" for x in nominal_loss]],
        ["5% loss, bounded drift", STRESS_RUNS, *[f"{x:g}" for x in stress_percent[2]]]])
    checks = nominal_checks + stress_checks
    save_table("compare_checks", ["quantity", "value"], [
        ["rank H_L(w_d) over all runs", sorted({int(r) for r, _ in checks})],
        ["Algorithm 1 stops", sum(s for _, s in checks)],
        ["windows per record", N_WINDOWS],
        ["parity residual dimension", q * L - d0],
        ["parity threshold chi2 quantile", f"{1 - ALPHA / N_WINDOWS:.6f}"]])
    figure(fractions)


if __name__ == "__main__":
    main()
