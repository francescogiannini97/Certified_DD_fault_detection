import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

from hankel import hankel
from output import BLUE, GREEN, ORANGE, TEXT_WIDTH, save_figure, save_raw, save_table
from records import q, record, rms, uniform_noise
from shear_frame import SAMPLING_RATE, lag, shear_frame
from weighting import ellipsoidal_W0, epsbar, falsified, weight, weighted_svd

SEED = 2027
m, n, L, N = 1, 6, 10, 4000
d0 = m * L + n
GAMMAS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
N_RUNS = 50
PROFILE_GAMMA = 1e-2


def realization(rng, system, gamma):
    wd = record(rng, system, N)
    eta = gamma * rms(wd)
    ed = uniform_noise(rng, N, eta)
    W0 = ellipsoidal_W0(eta)
    model = weighted_svd(wd + ed, W0, L, d0)
    WE = hankel(weight(ed, W0), L)
    U_B0 = np.linalg.svd(hankel(weight(wd, W0), L), full_matrices=False)[0][:, :d0]
    Ut1, St1 = model["Ut1"], model["St1"]
    F0 = WE @ model["Vt1"]
    X = Ut1 * St1 - F0
    sin_theta = (np.linalg.norm(X - Ut1 * np.sum(Ut1 * X, 0), axis=0)
                 / np.linalg.norm(X, axis=0))
    return dict(gap=np.linalg.norm(Ut1 - U_B0 @ (U_B0.T @ Ut1), 2),
                threshold=model["epsbar"] / St1[-1], s=np.r_[St1, model["s_next"]],
                norm_WE=np.linalg.norm(WE, 2), norm_F0=np.linalg.norm(F0, 2),
                sin_theta=sin_theta, rank=np.linalg.matrix_rank(hankel(wd, L)),
                falsified=falsified(model))


def run():
    rng = np.random.default_rng(SEED)
    system = shear_frame()
    return system, {g: [realization(rng, system, g) for _ in range(N_RUNS)] for g in GAMMAS}


def tables(system, results):
    eb = epsbar(L, N)
    save_table("model", ["quantity", "value"], [
        ["natural frequencies [Hz]", " ".join(f"{f:.4f}" for f in system["natural_frequencies"])],
        ["sampling rate [Hz]", SAMPLING_RATE], ["q", q], ["m", m],
        ["n", system["A"].shape[0]], ["lag", lag(system)], ["d0", d0],
        ["Hankel size", f"{q * L}x{N - L + 1}"], ["epsbar", f"{eb:.4f}"],
        ["delta", f"{np.sqrt(L):.4f}"],
        ["SNR [dB] gamma=0.1/0.3/1%",
         " ".join(f"{20 * np.log10(np.sqrt(3) / g):.2f}" for g in (1e-3, 3e-3, 1e-2))]])
    rows = []
    for g in GAMMAS:
        gap = np.array([r["gap"] for r in results[g]])
        threshold = np.array([r["threshold"] for r in results[g]])
        rows.append([f"{100 * g:g}", f"{np.mean(threshold < 1):.2f}",
                     int(np.all(gap < np.minimum(threshold, 1))),
                     f"{gap.max():.4e}", f"{np.minimum(threshold, 1).min():.4e}",
                     f"{gap.max() / np.minimum(threshold, 1).min():.4f}",
                     f"{np.median(threshold / gap):.1f}",
                     sorted({int(r['rank']) for r in results[g]}),
                     sum(bool(r["falsified"]) for r in results[g])])
    save_table("validation_gap", ["gamma [%]", "admissible fraction", "gap < threshold (all)",
                                  "largest gap", "smallest threshold",
                                  "largest gap / smallest threshold",
                                  "median threshold/gap", "rank H_L(w_d)",
                                  "runs with s_{d0+1} > epsbar"], rows)
    R = results[PROFILE_GAMMA]
    S = np.array([r["s"][:d0] for r in R])
    sin_theta = np.array([r["sin_theta"] for r in R])
    norm_F0 = np.array([r["norm_F0"] for r in R])
    norm_WE = np.array([r["norm_WE"] for r in R])
    F0_bound = norm_F0[:, None] / S
    save_table("validation_profile", ["quantity (gamma = 1%)", "value"], [
        ["median sin(theta_d0)/sin(theta_1)",
         f"{np.median(sin_theta[:, -1] / sin_theta[:, 0]):.3f}"],
        ["median s_1/s_d0", f"{np.median(S[:, 0] / S[:, -1]):.3f}"],
        ["max (||F0||/s_i)/sin(theta_i)", f"{(F0_bound / sin_theta).max():.3f}"],
        ["median (epsbar/s_i)/sin(theta_i)", f"{np.median((eb / S) / sin_theta):.1f}"],
        ["median ||WE||/epsbar", f"{np.median(norm_WE / eb):.4f}"],
        ["median ||WE||/||F0||", f"{np.median(norm_WE / norm_F0):.3f}"],
        ["median (||F0||/s_i)/sin(theta_i)", f"{np.median(F0_bound / sin_theta):.3f}"]])
    raw = []
    for g in GAMMAS:
        for k, r in enumerate(results[g]):
            raw.append([g, k, r["gap"], r["threshold"], r["norm_WE"], r["norm_F0"], int(r["rank"]),
                        *r["s"], *r["sin_theta"]])
    save_raw("validation_realizations",
             ["gamma", "run", "gap", "threshold", "norm_WE", "norm_F0", "rank",
              *[f"s_{i + 1}" for i in range(d0 + 1)],
              *[f"sin_theta_{i + 1}" for i in range(d0)]], raw)


def figure(results):
    eb = epsbar(L, N)
    fig, (a, b) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.75), layout="constrained")
    x = 100 * np.array(GAMMAS)
    jitter = np.exp(np.linspace(-0.08, 0.08, N_RUNS))
    largest_gap, smallest_threshold = [], []
    for xx, g in zip(x, GAMMAS):
        gap = np.array([r["gap"] for r in results[g]])
        threshold = np.minimum([r["threshold"] for r in results[g]], 1.0)
        a.scatter(xx * jitter, gap, s=5, color=BLUE, alpha=0.45, lw=0)
        a.scatter(xx * jitter, threshold, s=5, color=ORANGE, alpha=0.45, lw=0)
        largest_gap.append(gap.max())
        smallest_threshold.append(threshold.min())
    a.plot(x, largest_gap, "-", color=BLUE, lw=1.3)
    a.plot(x, smallest_threshold, "--", color=ORANGE, lw=1.3)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("noise level $\\gamma$ [%]")
    a.set_xticks([0.01, 0.1, 1, 10])
    a.set_xticklabels(["0.01", "0.1", "1", "10"])
    a.set_ylim(5e-6, 2)
    a.grid(alpha=0.3, lw=0.5, which="major")
    a.set_title("(a)")
    handles = [Line2D([], [], color=BLUE, marker="o", ms=3.5, ls="-", lw=1.3),
               Line2D([], [], color=ORANGE, marker="o", ms=3.5, ls="--", lw=1.3)]
    a.legend(handles, ["$\\mathrm{gap}_L(W\\mathfrak{B}^0|_L,\\mathcal{C}|_L)$",
                       "$\\bar\\varepsilon/s_{d_0}$"],
             loc="lower center", bbox_to_anchor=(0.5, 1.09), ncol=2, frameon=False,
             borderaxespad=0, columnspacing=1.0, handlelength=1.8)
    R = results[PROFILE_GAMMA]
    S = np.array([r["s"][:d0] for r in R])
    sin_theta = np.array([r["sin_theta"] for r in R])
    norm_F0 = np.array([r["norm_F0"] for r in R])
    i = np.arange(1, d0 + 1)
    jitter = np.linspace(-0.22, 0.22, len(S))
    for Y, color in [(eb / S, ORANGE), (norm_F0[:, None] / S, GREEN), (sin_theta, BLUE)]:
        b.scatter((i[None, :] + jitter[:, None]).ravel(), Y.ravel(), s=4, color=color,
                  alpha=0.35, lw=0)
        b.plot(i, np.median(Y, 0), "-", color=color, lw=1.3, marker="o", ms=3)
    b.set_yscale("log")
    b.set_xlabel("direction $i$")
    b.set_xticks([1, 4, 8, 12, 16])
    b.set_xlim(0.4, d0 + 0.6)
    b.grid(alpha=0.3, lw=0.5, which="major")
    b.set_title("(b) $\\gamma=1\\%$")
    handles = [Line2D([], [], color=c, marker="o", ms=3.5, ls="-", lw=1.3)
               for c in (BLUE, ORANGE, GREEN)]
    b.legend(handles, ["$\\sin\\theta_i$", "$\\bar\\varepsilon/s_i$", "$\\|F_0\\|/s_i$"],
             loc="lower center", bbox_to_anchor=(0.5, 1.09), ncol=3, frameon=False,
             borderaxespad=0, columnspacing=1.0, handlelength=1.8)
    save_figure(fig, "validation")


def main():
    system, results = run()
    tables(system, results)
    figure(results)


if __name__ == "__main__":
    main()
