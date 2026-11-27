import csv
import logging
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "figures")
RESULTS = os.path.join(HERE, "results")
TEXT_WIDTH = 390 / 72.27
BLUE, ORANGE, GREEN, GREY = "#2b6cb0", "#e8833a", "#2f8a4c", "#555555"

logging.getLogger("fontTools").setLevel(logging.ERROR)

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 10,
                     "axes.labelsize": 10, "axes.titlesize": 10, "xtick.labelsize": 9,
                     "ytick.labelsize": 9, "legend.fontsize": 9, "pdf.fonttype": 42})


def save_figure(fig, name):
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, name + ".pdf"))
    plt.close(fig)


def save_table(name, header, rows):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name + ".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print_table(name, header, rows)


def save_raw(name, header, rows):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name + ".csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows([[repr(float(x)) for x in row] for row in rows])


def print_table(name, header, rows):
    widths = [max(len(str(x)) for x in column) for column in zip(header, *rows)]
    print(f"\n{name}")
    for row in [header, *rows]:
        print("  ".join(str(x).ljust(w) for x, w in zip(row, widths)))
