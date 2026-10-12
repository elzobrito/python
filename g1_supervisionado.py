"""01–05 · Supervisionado — implementações em numpy."""
import numpy as np
from dados import iris, split, padroniza, acc, softmax, rbf, mostrar
from arvores import crescer_class, pred_class, n_folhas


def regressao_linear():
    mostrar(1, "Regressão Linear", "Ajusta a reta que minimiza o erro quadrático (prevê números).")
    xs = np.arange(10.0)
    ys = 2 * xs + 1
    X = np.c_[np.ones(10), xs]
    b, coef = np.linalg.lstsq(X, ys, rcond=None)[0]
    print(f"Out: coef: [{coef:.1f}] b: {b:.1f}")


def regressao_logistica():
    mostrar(2, "Regressão Logística", "Estima a probabilidade de cada classe com a função logística.")
    X, y = iris()
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr, Xte = padroniza(Xtr, Xte)
    n, d = Xtr.shape
    k = int(y.max()) + 1
    Y = np.eye(k)[ytr]
    Xb = np.c_[np.ones(n), Xtr]
    W = np.zeros((d + 1, k))
    for _ in range(800):
        p = softmax(Xb @ W)
        W -= 0.4 * Xb.T @ (p - Y) / n
    yhat = softmax(np.c_[np.ones(len(Xte)), Xte] @ W).argmax(1)
    print(f"Out: acurácia: {round(acc(yhat, yte), 3)}")


def arvore_decisao():
    mostrar(3, "Árvore de Decisão", "Divide os dados com perguntas do tipo 'atributo < limite?'.")
    X, y = iris()
    Xtr, Xte, ytr, yte = split(X, y)
    k = int(y.max()) + 1
    arv = crescer_class(Xtr, ytr, 0, max_depth=3, n_classes=k)
    print(f"Out: acurácia: {round(acc(pred_class(arv, Xte), yte), 3)}")
    print(f"     folhas: {n_folhas(arv)}")


def random_forest():
    mostrar(4, "Random Forest", "Vota entre muitas árvores treinadas em amostras diferentes.")
    X, y = iris()
    Xtr, Xte, ytr, yte = split(X, y)
    k = int(y.max()) + 1
    rng = np.random.default_rng(0)
    n, d = Xtr.shape
    mf = max(1, int(np.sqrt(d)))
    votos = np.zeros((len(Xte), k))
    for _ in range(40):
        idx = rng.integers(0, n, n)
        arv = crescer_class(Xtr[idx], ytr[idx], 0, 4, k, max_features=mf, rng=rng)
        votos[np.arange(len(Xte)), pred_class(arv, Xte)] += 1
    print(f"Out: acurácia: {round(acc(votos.argmax(1), yte), 3)}")


def svm():
    mostrar(5, "SVM", "Busca a fronteira com a maior margem entre as classes.")
    X, y = iris()
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr, Xte = padroniza(Xtr, Xte)
    gamma = 1.0 / (Xtr.shape[1] * Xtr.var())
    classes = np.unique(ytr)
    K = rbf(Xtr, Xtr, gamma) + 0.1 * np.eye(len(Xtr))
    alphas = []
    for c in classes:
        yy = np.where(ytr == c, 1.0, -1.0)
        alphas.append(np.linalg.solve(K, yy))
    Kt = rbf(Xte, Xtr, gamma)
    scores = np.stack([Kt @ a for a in alphas], 1)
    n_sv = int(sum((np.abs(a) > 0.05 * np.abs(a).max()).sum() for a in alphas) / len(classes))
    print(f"Out: acurácia: {round(acc(classes[scores.argmax(1)], yte), 3)}")
    print(f"     vetores de suporte: {n_sv}")


def rodar():
    print("\n— Supervisionado · iris —")
    regressao_linear()
    regressao_logistica()
    arvore_decisao()
    random_forest()
    svm()


if __name__ == "__main__":
    rodar()
