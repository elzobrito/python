"""Conjuntos de dados e helpers — só numpy."""
from pathlib import Path
import numpy as np

_DIR = Path(__file__).resolve().parent


def _sklearn_csv(nome):
    dados = np.loadtxt(_DIR / nome, delimiter=",", skiprows=1)
    return dados[:, :-1].astype(float), dados[:, -1].astype(int)


def iris():
    return _sklearn_csv("iris.csv")


def cancer():
    return _sklearn_csv("breast_cancer.csv")


def blobs(n=300, centros=3, std=0.6, d=2, seed=0):
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, centros, endpoint=False)
    c = np.stack([4.0 * np.cos(ang), 4.0 * np.sin(ang)], 1)
    if d > 2:
        c = np.c_[c, rng.normal(0, 0.1, size=(centros, d - 2))]
    y = np.repeat(np.arange(centros), n // centros)
    if len(y) < n:
        y = np.r_[y, np.zeros(n - len(y), dtype=int)]
    X = c[y[:n]] + rng.normal(0, std, size=(n, d))
    return X, y[:n]


def split(X, y, teste=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    nte = max(1, int(len(y) * teste))
    te, tr = idx[:nte], idx[nte:]
    return X[tr], X[te], y[tr], y[te]


def padroniza(Xtr, Xte=None):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    if Xte is None:
        return (Xtr - mu) / sd
    return (Xtr - mu) / sd, (Xte - mu) / sd


def acc(yhat, y):
    return float((np.asarray(yhat) == np.asarray(y)).mean())


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def softmax(z, eixo=-1):
    z = z - z.max(axis=eixo, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=eixo, keepdims=True)


def rbf(A, B, gamma=0.2):
    aa = np.einsum("ij,ij->i", A, A)[:, None]
    bb = np.einsum("ij,ij->i", B, B)[None, :]
    return np.exp(-gamma * np.maximum(aa + bb - 2 * A @ B.T, 0.0))


def dist2(A, B=None):
    if B is None:
        B = A
    aa = np.einsum("ij,ij->i", A, A)[:, None]
    bb = np.einsum("ij,ij->i", B, B)[None, :]
    return np.maximum(aa + bb - 2 * A @ B.T, 0.0)


def mostrar(num, nome, desc):
    print(f"\n== {num:02d} {nome}")
    print(desc)
