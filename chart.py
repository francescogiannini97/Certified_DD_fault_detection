import numpy as np
import matplotlib.pyplot as plt

import fd_monte_carlo as mc
from output import BLUE, ORANGE, TEXT_WIDTH, save_figure, save_table
from shear_frame import SAMPLING_RATE

GAMMA, DELTA, RUN = 1e-3, 0.03, 0


def main(results=None):
    distances = mc.load_distances() if results is None else results["distances"]
    record = distances[mc.GAMMAS.index(GAMMA), mc.DELTAS.index(DELTA), RUN]
    save_table("chart", ["test", "certified windows", "fraction"],
               [[name, int((record[:, k] > 1).sum()), f"{(record[:, k] > 1).mean():.4f}"]
                for k, name in enumerate(mc.TESTS)])
    t = np.arange(record.shape[0]) / SAMPLING_RATE
    fig, a = plt.subplots(1, 1, figsize=(0.8 * TEXT_WIDTH, 2.35), layout="constrained")
    a.plot(t, record[:, 3], color=BLUE, lw=1.2, label="anisotropic (ellipsoidal)")
    a.plot(t, record[:, 0], color=ORANGE, lw=1.4, label="gap-based")
    a.axhline(1, color="k", lw=0.9, ls=":", label="distance $=\\delta$")
    a.set_xlabel("window start time [s]")
    a.grid(alpha=0.3, lw=0.5)
    a.set_xlim(0, t[-1])
    a.set_ylim(-0.3, np.ceil(record[:, 3].max()) + 0.3)
    a.set_ylabel("$\\mathrm{dist}(W\\tilde w,\\,\\cdot\\,)/\\delta$")
    fig.legend(loc="outside upper center", ncol=3, frameon=False, columnspacing=0.9,
               handlelength=1.6, handletextpad=0.5)
    save_figure(fig, "chart")


if __name__ == "__main__":
    main()
