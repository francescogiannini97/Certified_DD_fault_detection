import numpy as np
from scipy.linalg import expm, solve

MASSES = [6.0, 6.0, 6.0]
STIFFNESSES = [2.0e5, 2.0e5, 2.0e5]
DAMPING_RATIO = 0.02
SAMPLING_RATE = 320.0


def shear_frame(first_story_loss=0.0):
    stiffnesses = np.r_[STIFFNESSES[0] * (1 - first_story_loss), STIFFNESSES[1:]]
    n_floors = len(MASSES)
    M = np.diag(MASSES)
    K = np.zeros((n_floors, n_floors))
    for i, k in enumerate(stiffnesses):
        K[i, i] += k
        if i > 0:
            K[i - 1, i - 1] += k
            K[i - 1, i] -= k
            K[i, i - 1] -= k
    omega_squared, _ = np.linalg.eig(solve(M, K))
    omega = np.sort(np.sqrt(omega_squared.real))
    a0, a1 = np.linalg.solve(
        np.array([[1 / (2 * omega[0]), omega[0] / 2], [1 / (2 * omega[1]), omega[1] / 2]]),
        [DAMPING_RATIO, DAMPING_RATIO],
    )
    C_damping = a0 * M + a1 * K
    M_inv = np.linalg.inv(M)
    Ac = np.block([[np.zeros((n_floors, n_floors)), np.eye(n_floors)],
                   [-M_inv @ K, -M_inv @ C_damping]])
    Bc = np.vstack([np.zeros((n_floors, 1)), -np.ones((n_floors, 1))])
    Cc = np.hstack([-M_inv @ K, -M_inv @ C_damping])
    n = 2 * n_floors
    zoh = expm(np.block([[Ac, Bc], [np.zeros((1, n + 1))]]) * (1 / SAMPLING_RATE))
    return dict(A=zoh[:n, :n], B=zoh[:n, n:], C=Cc, natural_frequencies=omega / (2 * np.pi))


def simulate(system, u, x0=None):
    A, B, C = system["A"], system["B"], system["C"]
    x = np.zeros(A.shape[0]) if x0 is None else x0.copy()
    y = []
    for t in range(len(u)):
        y.append(C @ x)
        x = A @ x + B @ u[t]
    return np.array(y), x


def lag(system):
    A, C = system["A"], system["C"]
    n = A.shape[0]
    blocks = [C]
    while np.linalg.matrix_rank(np.vstack(blocks)) < n:
        blocks.append(blocks[-1] @ A)
    return len(blocks)
