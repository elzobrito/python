"""16–20 · Anomalias e outros — implementações em numpy."""
import numpy as np
from dados import iris, blobs, padroniza, mostrar
from g3_nao_supervisionado import kmeans_pp_init, kmeans_fit


def kmeans_plusplus():
    mostrar(16, "k-Means++", "Inicialização do k-Means: centros iniciais espalhados.")
    X, _ = blobs()
    rng = np.random.default_rng(0)
    centros = kmeans_pp_init(X, 3, rng)
    print("Out: centros iniciais:", np.round(centros, 1).tolist())


def _c_fator(n):
    if n <= 1:
        return 0.0
    return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n


def _itree(X, h, hmax, rng):
    if h >= hmax or len(X) <= 1:
        return ("folha", len(X))
    j = int(rng.integers(X.shape[1]))
    lo, hi = float(X[:, j].min()), float(X[:, j].max())
    if lo == hi:
        return ("folha", len(X))
    t = float(rng.uniform(lo, hi))
    m = X[:, j] < t
    return ("no", j, t, _itree(X[m], h + 1, hmax, rng), _itree(X[~m], h + 1, hmax, rng))


def _caminho(no, x, h=0):
    if no[0] == "folha":
        return h + _c_fator(no[1])
    _, j, t, esq, dire = no
    return _caminho(esq if x[j] < t else dire, x, h + 1)


def isolation_forest():
    mostrar(17, "Isolation Forest", "Isola pontos com cortes aleatórios: anomalia isola rápido.")
    Xb, _ = blobs()
    Xo = np.vstack([Xb, [[10, 10], [-8, 12]]])
    rng = np.random.default_rng(0)
    n, n_arv, psi = len(Xo), 40, 64
    hmax = int(np.ceil(np.log2(psi)))
    florestas = []
    for _ in range(n_arv):
        idx = rng.integers(0, n, psi)
        florestas.append(_itree(Xo[idx], 0, hmax, rng))
    cn = _c_fator(psi)
    scores = np.array([
        2 ** (-np.mean([_caminho(t, x) for t in florestas]) / cn)
        for x in Xo
    ])
    limiar = np.quantile(scores, 0.99)
    pred = np.where(scores >= limiar, -1, 1)
    print(f"Out: anomalias: {int((pred == -1).sum())} | 2 estranhos: {pred[-2:].tolist()}")


def autoencoder():
    mostrar(18, "Autoencoder", "Comprime e reconstrói; erro alto indica algo fora do padrão.")
    X, _ = iris()
    X = padroniza(X)
    rng = np.random.default_rng(0)
    n, d = X.shape
    W1 = rng.normal(0, 0.2, (d, 2))
    b1 = np.zeros(2)
    W2 = rng.normal(0, 0.2, (2, d))
    b2 = np.zeros(d)
    lr = 0.05
    for _ in range(2500):
        h = np.tanh(X @ W1 + b1)
        y = h @ W2 + b2
        e = y - X
        gW2 = h.T @ e / n
        gb2 = e.mean(0)
        dh = (e @ W2.T) * (1 - h ** 2)
        gW1 = X.T @ dh / n
        gb1 = dh.mean(0)
        W2 -= lr * gW2
        b2 -= lr * gb2
        W1 -= lr * gW1
        b1 -= lr * gb1
    h = np.tanh(X @ W1 + b1)
    y = h @ W2 + b2
    erro = float(((y - X) ** 2).mean())
    print(f"Out: erro médio de reconstrução: {round(erro, 3)}")


def gmm():
    mostrar(19, "Mistura Gaussiana (GMM)", "Modela os dados como soma de gaussianas (grupos suaves).")
    X, _ = blobs()
    k = 3
    c, rot = kmeans_fit(X, k, n_init=5)
    pi = np.bincount(rot, minlength=k).astype(float)
    pi /= pi.sum()
    mu = c.copy()
    cov = np.stack([np.cov(X[rot == j].T) + 1e-3 * np.eye(2) if np.any(rot == j) else np.eye(2) for j in range(k)])
    for _ in range(40):
        logp = np.empty((len(X), k))
        for j in range(k):
            diff = X - mu[j]
            try:
                L = np.linalg.cholesky(cov[j])
                logdet = 2 * np.log(np.diag(L)).sum()
                sol = np.linalg.solve(L, diff.T)
                quad = (sol ** 2).sum(0)
            except np.linalg.LinAlgError:
                logdet = np.log(np.linalg.det(cov[j]) + 1e-12)
                quad = np.einsum("ni,ij,nj->n", diff, np.linalg.pinv(cov[j]), diff)
            logp[:, j] = np.log(pi[j] + 1e-12) - 0.5 * (logdet + quad + 2 * np.log(2 * np.pi))
        logp -= logp.max(1, keepdims=True)
        resp = np.exp(logp)
        resp /= resp.sum(1, keepdims=True)
        Nk = resp.sum(0) + 1e-12
        pi = Nk / len(X)
        mu = (resp.T @ X) / Nk[:, None]
        for j in range(k):
            d = X - mu[j]
            cov[j] = (d * resp[:, j:j + 1]).T @ d / Nk[j] + 1e-3 * np.eye(2)
    print("Out: pesos:", np.round(pi, 2).tolist())
    print("     P(1º ponto):", np.round(resp[:1], 2).tolist())


def genetico():
    mostrar(20, "Algoritmo Genético", "Seleção + cruzamento + mutação. Exemplo: maximizar 1s em 16 bits.")
    rng = np.random.default_rng(0)
    P = rng.integers(0, 2, (20, 16))
    for _ in range(40):
        pais = P[np.argsort(P.sum(1))[-10:]]
        mix = rng.random((10, 16)) < 0.5
        filhos = np.where(mix, pais, pais[::-1])
        filhos ^= rng.random((10, 16)) < 0.05
        P = np.vstack([pais, filhos])
    print(f"Out: melhor: {int(P.sum(1).max())} de 16 bits")


def rodar():
    print("\n— Anomalias e outros · blobs, iris e numpy —")
    kmeans_plusplus()
    isolation_forest()
    autoencoder()
    gmm()
    genetico()


if __name__ == "__main__":
    rodar()
