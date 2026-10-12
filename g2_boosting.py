"""06–10 · Probabilísticos e boosting — implementações em numpy."""
import numpy as np
from dados import cancer, split, padroniza, acc, sigmoid, mostrar
from arvores import crescer_class, pred_class, crescer_reg, pred_reg, crescer_xgb, pred_reg as pred_folha


def knn():
    mostrar(6, "k-NN", "Classifica pela maioria dos k vizinhos mais próximos.")
    X, y = cancer()
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr, Xte = padroniza(Xtr, Xte)
    d = ((Xte[:, None, :] - Xtr[None, :, :]) ** 2).sum(-1)
    viz = ytr[np.argsort(d, axis=1)[:, :5]]
    yhat = np.array([np.bincount(v).argmax() for v in viz])
    print(f"Out: acurácia: {round(acc(yhat, yte), 3)}")


def naive_bayes():
    mostrar(7, "Naive Bayes", "Aplica Bayes supondo atributos independentes entre si.")
    X, y = cancer()
    Xtr, Xte, ytr, yte = split(X, y)
    classes = np.unique(ytr)
    logp = []
    for c in classes:
        Xc = Xtr[ytr == c]
        mu, var = Xc.mean(0), Xc.var(0) + 1e-9
        lp = np.log((ytr == c).mean() + 1e-12)
        ll = -0.5 * np.sum(np.log(2 * np.pi * var) + (Xte - mu) ** 2 / var, 1)
        logp.append(lp + ll)
    yhat = classes[np.stack(logp, 1).argmax(1)]
    print(f"Out: acurácia: {round(acc(yhat, yte), 3)}")


def gradient_boosting():
    mostrar(8, "Gradient Boosting", "Cada árvore nova corrige o erro (gradiente) das anteriores.")
    X, y = cancer()
    Xtr, Xte, ytr, yte = split(X, y)
    ytr = ytr.astype(float)
    F = np.full(len(ytr), np.log((ytr.mean() + 1e-6) / (1 - ytr.mean() + 1e-6)))
    Ft = np.zeros(len(Xte))
    lr = 0.1
    for _ in range(40):
        p = sigmoid(F)
        r = ytr - p
        arv = crescer_reg(Xtr, r, 0, max_depth=2)
        F += lr * pred_reg(arv, Xtr)
        Ft += lr * pred_reg(arv, Xte)
    yhat = (sigmoid(Ft) >= 0.5).astype(int)
    print(f"Out: acurácia: {round(acc(yhat, yte), 3)}")


def adaboost():
    mostrar(9, "AdaBoost", "Dá mais peso aos exemplos que os modelos anteriores erraram.")
    X, y = cancer()
    Xtr, Xte, ytr, yte = split(X, y)
    n = len(ytr)
    w = np.full(n, 1.0 / n)
    pred_te = np.zeros(len(Xte))
    k = 2
    for _ in range(30):
        arv = crescer_class(Xtr, ytr, 0, 1, k, pesos=w)
        yhat = pred_class(arv, Xtr)
        err = float(np.dot(w, yhat != ytr))
        err = min(max(err, 1e-9), 1 - 1e-9)
        alfa = 0.5 * np.log((1 - err) / err)
        w *= np.exp(alfa * np.where(yhat == ytr, -1.0, 1.0))
        w /= w.sum()
        pred_te += alfa * np.where(pred_class(arv, Xte) == 1, 1.0, -1.0)
    yhat = (pred_te >= 0).astype(int)
    print(f"Out: acurácia: {round(acc(yhat, yte), 3)}")


def xgboost_mini():
    mostrar(10, "XGBoost", "Gradient boosting otimizado, com regularização.")
    X, y = cancer()
    Xtr, Xte, ytr, yte = split(X, y)
    ytr = ytr.astype(float)
    F = np.zeros(len(ytr))
    Ft = np.zeros(len(Xte))
    lam, lr = 1.0, 0.3
    for _ in range(40):
        p = sigmoid(F)
        g, h = p - ytr, p * (1 - p) + 1e-6
        arv = crescer_xgb(Xtr, g, h, 0, max_depth=3, lam=lam)
        F += lr * pred_folha(arv, Xtr)
        Ft += lr * pred_folha(arv, Xte)
    yhat = (sigmoid(Ft) >= 0.5).astype(int)
    print(f"Out: acurácia: {round(acc(yhat, yte), 3)}")


def rodar():
    print("\n— Probabilísticos e Boosting · breast_cancer —")
    knn()
    naive_bayes()
    gradient_boosting()
    adaboost()
    xgboost_mini()


if __name__ == "__main__":
    rodar()
