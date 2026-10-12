"""Árvores de classificação, regressão e ganho estilo XGBoost."""
import numpy as np


class No:
    __slots__ = ("j", "t", "esq", "dir", "valor", "folha")

    def __init__(self, folha=False, valor=None, j=None, t=None, esq=None, dir=None):
        self.folha = folha
        self.valor = valor
        self.j = j
        self.t = t
        self.esq = esq
        self.dir = dir


def _gini(cont):
    n = float(cont.sum())
    if n <= 0:
        return 0.0
    p = cont / n
    return 1.0 - float(p @ p)


def _feats(d, max_features, rng):
    feats = np.arange(d)
    if max_features is not None and max_features < d:
        feats = rng.choice(d, size=max(1, max_features), replace=False)
    return feats


def melhor_corte_class(X, y, feats, n_classes, pesos):
    n = len(y)
    melhor = None
    pai = np.bincount(y, weights=pesos, minlength=n_classes)
    gini_pai = _gini(pai)
    wtot = float(pesos.sum())
    for j in feats:
        ordem = np.argsort(X[:, j], kind="mergesort")
        xs, ys, ws = X[ordem, j], y[ordem], pesos[ordem]
        esq = np.zeros(n_classes)
        dire = pai.copy()
        w_esq = 0.0
        for i in range(1, n):
            c, w = int(ys[i - 1]), float(ws[i - 1])
            esq[c] += w
            dire[c] -= w
            w_esq += w
            if xs[i] == xs[i - 1] or w_esq <= 0 or (wtot - w_esq) <= 0:
                continue
            g = gini_pai - (w_esq / wtot) * _gini(esq) - ((wtot - w_esq) / wtot) * _gini(dire)
            if melhor is None or g > melhor[0]:
                melhor = (g, int(j), 0.5 * (xs[i - 1] + xs[i]))
    return melhor


def crescer_class(X, y, profundidade, max_depth, n_classes, max_features=None, rng=None, pesos=None, min_folha=1):
    if pesos is None:
        pesos = np.ones(len(y))
    cont = np.bincount(y, weights=pesos, minlength=n_classes)
    valor = int(cont.argmax())
    if profundidade >= max_depth or len(y) <= min_folha * 2 or cont.max() >= pesos.sum() - 1e-12:
        return No(folha=True, valor=valor)
    corte = melhor_corte_class(X, y, _feats(X.shape[1], max_features, rng), n_classes, pesos)
    if corte is None or corte[0] <= 1e-12:
        return No(folha=True, valor=valor)
    _, j, t = corte
    m = X[:, j] <= t
    if m.all() or (~m).all():
        return No(folha=True, valor=valor)
    return No(
        folha=False, j=j, t=t, valor=valor,
        esq=crescer_class(X[m], y[m], profundidade + 1, max_depth, n_classes, max_features, rng, pesos[m], min_folha),
        dir=crescer_class(X[~m], y[~m], profundidade + 1, max_depth, n_classes, max_features, rng, pesos[~m], min_folha),
    )


def pred_class_um(no, x):
    while not no.folha:
        no = no.esq if x[no.j] <= no.t else no.dir
    return no.valor


def pred_class(no, X):
    return np.fromiter((pred_class_um(no, x) for x in X), dtype=int, count=len(X))


def n_folhas(no):
    if no.folha:
        return 1
    return n_folhas(no.esq) + n_folhas(no.dir)


def melhor_corte_reg(X, r, feats):
    n = len(r)
    sse_pai = float(((r - r.mean()) ** 2).sum())
    melhor = None
    sum_tot = float(r.sum())
    sumsq_tot = float(r @ r)
    for j in feats:
        ordem = np.argsort(X[:, j], kind="mergesort")
        xs, rr = X[ordem, j], r[ordem]
        sum_esq = 0.0
        sumsq_esq = 0.0
        for i in range(1, n):
            sum_esq += rr[i - 1]
            sumsq_esq += rr[i - 1] ** 2
            if xs[i] == xs[i - 1]:
                continue
            nL, nR = i, n - i
            sse_L = sumsq_esq - sum_esq ** 2 / nL
            sse_R = (sumsq_tot - sumsq_esq) - (sum_tot - sum_esq) ** 2 / nR
            ganho = sse_pai - sse_L - sse_R
            if melhor is None or ganho > melhor[0]:
                melhor = (ganho, int(j), 0.5 * (xs[i - 1] + xs[i]))
    return melhor


def crescer_reg(X, r, profundidade, max_depth, min_folha=2):
    valor = float(r.mean()) if len(r) else 0.0
    if profundidade >= max_depth or len(r) <= min_folha * 2:
        return No(folha=True, valor=valor)
    corte = melhor_corte_reg(X, r, np.arange(X.shape[1]))
    if corte is None or corte[0] <= 1e-12:
        return No(folha=True, valor=valor)
    _, j, t = corte
    m = X[:, j] <= t
    if m.all() or (~m).all():
        return No(folha=True, valor=valor)
    return No(
        folha=False, j=j, t=t, valor=valor,
        esq=crescer_reg(X[m], r[m], profundidade + 1, max_depth, min_folha),
        dir=crescer_reg(X[~m], r[~m], profundidade + 1, max_depth, min_folha),
    )


def melhor_corte_xgb(X, g, h, feats, lam=1.0, gamma=0.0):
    G, H = float(g.sum()), float(h.sum())
    melhor = None
    for j in feats:
        ordem = np.argsort(X[:, j], kind="mergesort")
        xs, gs, hs = X[ordem, j], g[ordem], h[ordem]
        GL = HL = 0.0
        for i in range(1, len(g)):
            GL += gs[i - 1]
            HL += hs[i - 1]
            if xs[i] == xs[i - 1]:
                continue
            GR, HR = G - GL, H - HL
            if HL < 1e-8 or HR < 1e-8:
                continue
            ganho = 0.5 * (GL ** 2 / (HL + lam) + GR ** 2 / (HR + lam) - G ** 2 / (H + lam)) - gamma
            if melhor is None or ganho > melhor[0]:
                melhor = (ganho, int(j), 0.5 * (xs[i - 1] + xs[i]))
    return melhor


def crescer_xgb(X, g, h, profundidade, max_depth, lam=1.0, gamma=0.0):
    G, H = float(g.sum()), float(h.sum())
    w = -G / (H + lam)
    if profundidade >= max_depth or len(g) < 4:
        return No(folha=True, valor=w)
    corte = melhor_corte_xgb(X, g, h, np.arange(X.shape[1]), lam, gamma)
    if corte is None or corte[0] <= 1e-12:
        return No(folha=True, valor=w)
    _, j, t = corte
    m = X[:, j] <= t
    if m.all() or (~m).all():
        return No(folha=True, valor=w)
    return No(
        folha=False, j=j, t=t, valor=w,
        esq=crescer_xgb(X[m], g[m], h[m], profundidade + 1, max_depth, lam, gamma),
        dir=crescer_xgb(X[~m], g[~m], h[~m], profundidade + 1, max_depth, lam, gamma),
    )


def pred_reg_um(no, x):
    while not no.folha:
        no = no.esq if x[no.j] <= no.t else no.dir
    return float(no.valor)


def pred_reg(no, X):
    return np.fromiter((pred_reg_um(no, x) for x in X), dtype=float, count=len(X))
