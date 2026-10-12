"""11–15 · Não supervisionado — implementações em numpy."""
import numpy as np
from dados import iris, blobs, dist2, mostrar


def kmeans_pp_init(X, k, rng):
    n = len(X)
    centros = np.empty((k, X.shape[1]))
    centros[0] = X[rng.integers(n)]
    d2 = dist2(X, centros[:1]).ravel()
    for t in range(1, k):
        p = d2 / (d2.sum() + 1e-12)
        centros[t] = X[rng.choice(n, p=p)]
        d2 = np.minimum(d2, dist2(X, centros[t : t + 1]).ravel())
    return centros


def kmeans_fit(X, k, seed=0, n_init=10, max_iter=50):
    rng = np.random.default_rng(seed)
    melhor, melhor_inercia = None, np.inf
    for s in range(n_init):
        c = kmeans_pp_init(X, k, np.random.default_rng(int(rng.integers(1 << 30))))
        rot = np.zeros(len(X), dtype=int)
        for _ in range(max_iter):
            d = dist2(X, c)
            rot = d.argmin(1)
            novo = np.stack([
                X[rot == j].mean(0) if np.any(rot == j) else c[j]
                for j in range(k)
            ])
            if np.allclose(novo, c):
                break
            c = novo
        inercia = float(dist2(X, c)[np.arange(len(X)), rot].sum())
        if inercia < melhor_inercia:
            melhor, melhor_inercia = (c, rot), inercia
    return melhor


def k_means():
    mostrar(11, "k-Means", "Agrupa em k grupos, aproximando cada ponto do centro mais perto.")
    X, _ = blobs()
    _, rot = kmeans_fit(X, 3)
    print("Out: pontos por grupo:", np.bincount(rot).tolist())


def aglomerativo():
    mostrar(12, "Hierárquico (Aglomerativo)", "Junta os grupos mais próximos, passo a passo, até sobrar k.")
    X, _ = blobs()
    n, k = len(X), 3
    D = np.sqrt(dist2(X))
    np.fill_diagonal(D, np.inf)
    ativo = np.ones(n, dtype=bool)
    tam = np.ones(n, dtype=int)
    rot = np.arange(n)
    for _ in range(n - k):
        M = D.copy()
        M[~ativo] = np.inf
        M[:, ~ativo] = np.inf
        i, j = np.unravel_index(np.argmin(M), M.shape)
        if i > j:
            i, j = j, i
        ni, nj = int(tam[i]), int(tam[j])
        for m in np.flatnonzero(ativo):
            if m == i or m == j:
                continue
            d = (ni * D[i, m] + nj * D[j, m]) / (ni + nj)
            D[i, m] = D[m, i] = d
        rot[rot == j] = i
        tam[i] = ni + nj
        ativo[j] = False
        D[j, :] = D[:, j] = np.inf
        D[i, i] = np.inf
    mapa = {c: t for t, c in enumerate(np.unique(rot))}
    rot = np.array([mapa[c] for c in rot])
    print("Out: pontos por grupo:", np.bincount(rot).tolist())


def dbscan():
    mostrar(13, "DBSCAN", "Forma grupos por densidade e marca pontos isolados como ruído.")
    X, _ = blobs()
    eps2, min_pts = 0.8 ** 2, 5
    n = len(X)
    D = dist2(X)
    rot = -np.ones(n, dtype=int)
    visitado = np.zeros(n, dtype=bool)
    cid = 0
    for i in range(n):
        if visitado[i]:
            continue
        visitado[i] = True
        viz = np.flatnonzero(D[i] <= eps2)
        if len(viz) < min_pts:
            continue
        rot[i] = cid
        fila = list(viz)
        seen = set(viz.tolist())
        while fila:
            j = fila.pop()
            if not visitado[j]:
                visitado[j] = True
                vj = np.flatnonzero(D[j] <= eps2)
                if len(vj) >= min_pts:
                    for u in vj:
                        if u not in seen:
                            seen.add(int(u))
                            fila.append(int(u))
            if rot[j] < 0:
                rot[j] = cid
        cid += 1
    print(f"Out: grupos: {cid} ruído: {int((rot == -1).sum())}")


def pca():
    mostrar(14, "PCA", "Projeta os dados nas direções de maior variância.")
    X, _ = iris()
    X = X - X.mean(0)
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    var = s ** 2
    var /= var.sum()
    print("Out: variância explicada:", np.round(var[:2], 3).tolist())


def tsne():
    mostrar(15, "t-SNE", "Mapeia para 2D preservando vizinhanças (para visualizar).")
    X, _ = iris()
    X = X - X.mean(0)
    rng = np.random.default_rng(0)
    n = len(X)
    D = dist2(X)
    np.fill_diagonal(D, 0.0)
    sig = np.empty(n)
    alvo = 30.0
    for i in range(n):
        lo, hi = 1e-4, 1e3
        d = D[i]
        for _ in range(35):
            mid = 0.5 * (lo + hi)
            p = np.exp(-d / (2 * mid * mid))
            p[i] = 0.0
            p = p / (p.sum() + 1e-12)
            perp = 2 ** float(-(p * np.log2(p + 1e-12)).sum())
            if perp > alvo:
                hi = mid
            else:
                lo = mid
        sig[i] = 0.5 * (lo + hi)
    P = np.exp(-D / (2 * sig[:, None] ** 2))
    np.fill_diagonal(P, 0.0)
    P /= P.sum(1, keepdims=True) + 1e-12
    P = np.maximum((P + P.T) / (2 * n), 1e-12)
    Y = rng.normal(0, 1e-4, size=(n, 2))
    iY = np.zeros_like(Y)
    for t in range(220):
        exag = 4.0 if t < 50 else 1.0
        mom = 0.5 if t < 50 else 0.8
        num = 1.0 / (1.0 + dist2(Y))
        np.fill_diagonal(num, 0.0)
        Q = np.maximum(num / (num.sum() + 1e-12), 1e-12)
        PQ = (exag * P - Q) * num
        grad = 4 * (PQ.sum(1)[:, None] * Y - PQ @ Y)
        iY = mom * iY - 120.0 * grad
        Y += iY
        Y -= Y.mean(0)
    print("Out: forma:", list(Y.shape), "1º ponto:", np.round(Y[0], 1).tolist())


def rodar():
    print("\n— Não supervisionado · blobs e iris —")
    k_means()
    aglomerativo()
    dbscan()
    pca()
    tsne()


if __name__ == "__main__":
    rodar()
