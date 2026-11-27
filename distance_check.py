import numpy as np

import fd_monte_carlo as mc
from cone_distance import LOG_TAU_RANGE, golden_section, phi
from output import save_table

REFERENCE_GRID = 6001
REFERENCE_ITER = 100
CHUNK = 50


def reference_distance(s, eps, c, r2):
    grid = np.linspace(-LOG_TAU_RANGE, LOG_TAU_RANGE, REFERENCE_GRID)
    values = phi(grid[:, None] * np.ones((1, c.shape[1])), s, eps, c, r2)
    interior = (values[1:-1] < values[:-2]) & (values[1:-1] <= values[2:])
    i, j = np.nonzero(interior)
    i = i + 1
    refined = phi(golden_section(lambda x: phi(x[None], s, eps, c[:, j], r2[j])[0],
                                 grid[i - 1], grid[i + 1], REFERENCE_ITER)[None],
                  s, eps, c[:, j], r2[j])[0]
    reference = np.sqrt(np.sum(c ** 2, 0) + r2)
    np.minimum.at(reference, j, refined)
    edge_minimum = values[-1] <= values[-2]
    return np.maximum(reference, 0), np.bincount(j, minlength=c.shape[1]), edge_minimum


def main(results=None):
    results = mc.run() if results is None else results
    algorithm, reference, n_minima, edge = [], [], [], []
    for item in results["outside_cones"]:
        for start in range(0, item["r2"].size, CHUNK):
            part = slice(start, start + CHUNK)
            ref, count, at_edge = reference_distance(item["St1"], item["epsbar"],
                                                     item["c"][:, part], item["r2"][part])
            reference.append(ref)
            n_minima.append(count)
            edge.append(at_edge)
        algorithm.append(item["distance"])
    algorithm, reference = np.concatenate(algorithm), np.concatenate(reference)
    n_minima, edge = np.concatenate(n_minima), np.concatenate(edge)
    relative = np.abs(algorithm - reference) / reference
    delta = mc.delta
    save_table("distance_check", ["quantity", "value"], [
        ["windows outside the anisotropic cones (ball + ellipsoidal)", algorithm.size],
        ["windows with exactly one interior local minimum of phi", int((n_minima == 1).sum())],
        ["windows with more than one interior local minimum", int((n_minima > 1).sum())],
        ["windows with no interior local minimum", int((n_minima == 0).sum())],
        ["windows with a minimum at the right end of the grid", int(edge.sum())],
        ["max relative difference algorithm vs reference", f"{relative.max():.3e}"],
        ["windows where algorithm < reference", int((algorithm < reference).sum())],
        ["decision changes (dist > delta)",
         int(((algorithm > delta) != (reference > delta)).sum())]])


if __name__ == "__main__":
    main()
