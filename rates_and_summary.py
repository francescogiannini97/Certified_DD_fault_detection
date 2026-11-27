import numpy as np
import matplotlib.pyplot as plt

import fd_monte_carlo as mc
from output import BLUE, ORANGE, TEXT_WIDTH, save_figure, save_table

STYLE = [("--", "o", ORANGE), ("--", "s", BLUE), ("-", "o", ORANGE), ("-", "s", BLUE)]
LEGEND_ORDER = [0, 2, 1, 3]


def certified(distances):
    return distances > 1


def smallest_delta(condition):
    hits = [D for D, ok in zip(mc.DELTAS, condition) if D > 0 and ok]
    return f"{100 * hits[0]:g}%" if hits else "--"


def summary_table(results):
    distances, info = results["distances"], results["info"]
    mean = {key: [np.mean([x[key] for x in info[g]]) for g in mc.GAMMAS]
            for key in info[mc.GAMMAS[0]][0]}
    columns = [f"gamma={100 * g:g}%" for g in mc.GAMMAS]
    rows = [["fbar = epsbar/s_d0", "ball", *[f"{x:.4f}" for x in mean["fbar_ball"]]],
            ["", "ellipsoidal", *[f"{x:.4f}" for x in mean["fbar_ellipsoidal"]]],
            ["s_1/s_d0", "ball", *[f"{x:.3f}" for x in mean["cond_ball"]]],
            ["", "ellipsoidal", *[f"{x:.3f}" for x in mean["cond_ellipsoidal"]]],
            ["||WE||/epsbar", "ellipsoidal", *[f"{x:.4f}" for x in mean["WE_ratio"]]]]
    names = [("isotropic (gap-based)", "ball"), ("anisotropic", "ball"),
             ("isotropic", "ellipsoidal"), ("anisotropic", "ellipsoidal")]
    for label, rule in [("every record", lambda d: certified(d).any(1).all(0)),
                        ("half of windows", lambda d: certified(d).mean((0, 1)) >= 0.5)]:
        for k in LEGEND_ORDER:
            rows.append([f"{label}: {names[k][0]}", names[k][1],
                         *[smallest_delta([rule(distances[a, b, :, :, k])
                                           for b in range(len(mc.DELTAS))])
                           for a in range(len(mc.GAMMAS))]])
    save_table("summary", ["quantity", "noise model", *columns], rows)
    checks = []
    for g in mc.GAMMAS:
        x = info[g]
        checks.append([f"{100 * g:g}", f"{max(r['fbar_ball'] for r in x):.4f}",
                       f"{max(r['fbar_ellipsoidal'] for r in x):.4f}",
                       f"{min(min(r['cond_ball'], r['cond_ellipsoidal']) for r in x):.3f}",
                       f"{max(max(r['cond_ball'], r['cond_ellipsoidal']) for r in x):.3f}",
                       sorted({int(r['rank']) for r in x}), sum(r["stops"] for r in x),
                       sum(r["falsified"] for r in x)])
    save_table("summary_checks", ["gamma [%]", "max fbar ball", "max fbar ellipsoidal",
                                  "min s_1/s_d0 (any model, run)",
                                  "max s_1/s_d0 (any model, run)", "rank H_L(w_d)",
                                  "Algorithm 1 stops", "runs with s_{d0+1} > epsbar"], checks)
    healthy = distances[:, 0]
    gap_based = certified(distances[:, :, :, :, 0])
    save_table("summary_counts", ["quantity", *columns], [
        ["false alarms (healthy windows certified, 4 tests)",
         *[int(certified(healthy[a]).sum()) for a in range(len(mc.GAMMAS))]],
        ["largest healthy distance / delta (4 tests)",
         *[f"{np.nanmax(healthy[a]):.3g}" for a in range(len(mc.GAMMAS))]],
        ["gap-based certified windows, all Delta",
         *[int(gap_based[a].sum()) for a in range(len(mc.GAMMAS))]],
        ["gap-based records with a certified window, all Delta",
         *[int(gap_based[a].any(2).sum()) for a in range(len(mc.GAMMAS))]],
        ["Delta of gap-based certified windows",
         *[" ".join(f"{100 * D:g}%" for b, D in enumerate(mc.DELTAS) if gap_based[a, b].any())
           or "--" for a in range(len(mc.GAMMAS))]],
        ["windows per (gamma, Delta)", *[distances.shape[2] * distances.shape[3]] * 3]])


def rates_table_and_figure(distances):
    rates = certified(distances).mean((2, 3))
    save_table("rates", ["gamma [%]", "Delta [%]", *mc.TESTS],
               [[f"{100 * g:g}", f"{100 * D:g}", *[f"{rates[a, b, k]:.4f}" for k in range(4)]]
                for a, g in enumerate(mc.GAMMAS) for b, D in enumerate(mc.DELTAS)])
    labels = ["isotropic, ball model (gap-based)", "anisotropic, ball model",
              "isotropic, ellipsoidal model", "anisotropic, ellipsoidal model"]
    fig, axes = plt.subplots(1, 3, figsize=(TEXT_WIDTH, 2.55), sharey=True, layout="constrained")
    x = 100 * np.array(mc.DELTAS)
    for a, (ax, g) in enumerate(zip(axes, mc.GAMMAS)):
        for k, (ls, marker, color) in enumerate(STYLE):
            ax.plot(x, rates[a, :, k], ls=ls, marker=marker, ms=3.5, color=color,
                    mfc="white" if k < 2 else color, lw=1.2, label=labels[k])
        ax.set_title(f"$\\gamma={100 * g:g}\\%$")
        ax.set_xlim(0, 30.5)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0, 10, 20, 30])
        ax.grid(alpha=0.3, lw=0.5)
    axes[0].set_ylabel("certified windows (fraction)")
    fig.supxlabel("first-story stiffness loss $\\Delta$ [%]", fontsize=10)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend([handles[i] for i in LEGEND_ORDER], [names[i] for i in LEGEND_ORDER],
               loc="outside upper center", ncol=2, frameon=False, columnspacing=1.2,
               handlelength=2.4)
    save_figure(fig, "rates")


def main(results=None):
    results = mc.run() if results is None else results
    summary_table(results)
    rates_table_and_figure(results["distances"])


if __name__ == "__main__":
    main()
