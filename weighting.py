import numpy as np

from hankel import hankel


def ball_W0(eta):
    return np.eye(len(eta)) / np.linalg.norm(eta)


def ellipsoidal_W0(eta):
    return np.diag(1 / eta) / np.sqrt(len(eta))


def weight(z, W0):
    return z @ W0.T


def epsbar(L, N):
    return np.sqrt(L * (N - L + 1))


def weighted_svd(wd_noisy, W0, L, d0):
    U, s, Vt = np.linalg.svd(hankel(weight(wd_noisy, W0), L), full_matrices=False)
    return dict(W0=W0, L=L, Ut1=U[:, :d0], Ut2=U[:, d0:], St1=s[:d0], s_next=s[d0],
                Vt1=Vt[:d0].T, epsbar=epsbar(L, len(wd_noisy)))


def no_certificate(model):
    return model["epsbar"] >= model["St1"][-1]


def falsified(model):
    return model["s_next"] > model["epsbar"]


def weighted_windows(model, w_noisy):
    return hankel(weight(w_noisy, model["W0"]), model["L"])


def coordinates(model, w_bar):
    c = model["Ut1"].T @ w_bar
    r = w_bar - model["Ut1"] @ c
    return c, np.sum(r ** 2, 0)
