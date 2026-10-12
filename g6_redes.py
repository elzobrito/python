"""26–30 · Redes neurais e sequências — implementações em numpy."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as janelas
from dados import iris, split, padroniza, acc, sigmoid, softmax, mostrar


def mlp():
    mostrar(26, "MLP / Rede neural (ANN)", "Camadas de neurônios com ativação não linear, treinadas por backprop.")
    X, y = iris()
    Xtr, Xte, ytr, yte = split(X, y)
    Xtr, Xte = padroniza(Xtr, Xte)
    rng = np.random.default_rng(0)
    n, d = Xtr.shape
    k, h = int(y.max()) + 1, 16
    Y = np.eye(k)[ytr]
    W1 = rng.normal(0, 0.4, (d, h))
    b1 = np.zeros(h)
    W2 = rng.normal(0, 0.4, (h, k))
    b2 = np.zeros(k)
    lr = 0.08
    for _ in range(1200):
        z1 = Xtr @ W1 + b1
        a1 = np.tanh(z1)
        p = softmax(a1 @ W2 + b2)
        e2 = (p - Y) / n
        gW2 = a1.T @ e2
        gb2 = e2.sum(0)
        e1 = (e2 @ W2.T) * (1 - a1 ** 2)
        gW1 = Xtr.T @ e1
        gb1 = e1.sum(0)
        W2 -= lr * gW2
        b2 -= lr * gb2
        W1 -= lr * gW1
        b1 -= lr * gb1
    hte = np.tanh(Xte @ W1 + b1)
    yhat = softmax(hte @ W2 + b2).argmax(1)
    print(f"Out: acurácia: {round(acc(yhat, yte), 3)}")


def cnn():
    mostrar(27, "CNN (convolução 2D)", "Desliza um filtro pela imagem para detectar padrões locais.")
    img = np.zeros((5, 5))
    img[:, 2:] = 1
    k = np.array([[-1.0, 0.0, 1.0]] * 3)
    out = (janelas(img, (3, 3)) * k).sum((2, 3))
    print("Out:", np.round(out[0], 2).tolist())


def rnn():
    mostrar(28, "RNN", "Lê a sequência passo a passo, levando um estado oculto h.")
    rng = np.random.default_rng(0)
    Wx, Wh = rng.normal(size=3), rng.normal(size=(3, 3)) * 0.5
    h = np.zeros(3)
    for x in [1.0, 0.0, -1.0]:
        h = np.tanh(Wx * x + Wh @ h)
    print("Out: estado final h:", np.round(h, 3).tolist())


def lstm():
    mostrar(29, "LSTM (célula)", "RNN com portões (esquecer, entrada, saída) e memória c.")
    rng = np.random.default_rng(0)
    W, h, c = rng.normal(size=(4, 3, 4)) * 0.5, np.zeros(3), np.zeros(3)
    for x in [1.0, 0.0, -1.0]:
        f, i, o, g = W @ np.r_[x, h]
        c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
        h = sigmoid(o) * np.tanh(c)
    print("Out: h:", np.round(h, 3).tolist(), "c:", np.round(c, 3).tolist())


def transformer():
    mostrar(30, "Transformer (atenção)", "Cada token pondera todos os outros: softmax(QKᵀ/√d)·V.")
    rng = np.random.default_rng(0)
    Q, K, V = rng.normal(size=(3, 4, 8))  # 4 tokens, d = 8
    A = softmax(Q @ K.T / np.sqrt(8), eixo=-1)
    print("Out: pesos do token 0:", np.round(A[0], 2).tolist())
    print("     saída:", (A @ V).shape)


def rodar():
    print("\n— Redes neurais e sequências · iris e numpy —")
    mlp()
    cnn()
    rnn()
    lstm()
    transformer()


if __name__ == "__main__":
    rodar()
