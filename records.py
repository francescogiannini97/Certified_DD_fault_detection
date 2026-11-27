import numpy as np

from shear_frame import simulate

TRANSIENT = 500
q = 4


def record(rng, system, length):
    _, x = simulate(system, rng.standard_normal((TRANSIENT, 1)))
    u = rng.standard_normal((length, 1))
    y, _ = simulate(system, u, x)
    return np.hstack([u, y])


def rms(w):
    return np.sqrt((w ** 2).mean(0))


def uniform_noise(rng, length, eta):
    return rng.uniform(-1, 1, (length, q)) * eta


def bounded_drift(rng, length, eta, sampling_rate):
    phase = rng.uniform(0, 2 * np.pi, (1, q))
    frequency = rng.uniform(0.2, 1.0, (1, q)) / sampling_rate
    t = np.arange(length)[:, None]
    return eta * np.sin(2 * np.pi * frequency * t + phase)

