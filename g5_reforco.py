"""21–25 · Aprendizado por reforço — versões didáticas em numpy."""
import numpy as np
from dados import softmax, mostrar


def passo(s, a):
    s2 = min(max(s + (1 if a else -1), 0), 4)
    return s2, float(s2 == 4), s2 == 4


def eps_greedy(Q, s, rng, eps=0.2):
    return int(rng.integers(2)) if rng.random() < eps else int(Q[s].argmax())


def rodar_corredor(Q, atualiza, n, rng):
    for _ in range(n):
        s, a, fim = 0, eps_greedy(Q, 0, rng), False
        while not fim:
            s2, r, fim = passo(s, a)
            a2 = eps_greedy(Q, s2, rng)
            atualiza(s, a, r, s2, a2, fim)
            s, a = s2, a2


class MiniRede:
    def __init__(self, rng):
        self.W1 = rng.normal(0, 0.5, (5, 8))
        self.W2 = rng.normal(0, 0.5, (8, 2))

    def q(self, x):
        self.h = np.tanh(np.eye(5)[x] @ self.W1)
        return self.h @ self.W2

    def treina(self, x, a, alvo, lr=0.05):
        e = self.q(x)[a] - alvo
        g = e * self.W2[:, a] * (1 - self.h ** 2)
        self.W2[:, a] -= lr * e * self.h
        self.W1 -= lr * np.outer(np.eye(5)[x], g)

    def copia(self):
        c = MiniRede.__new__(MiniRede)
        c.W1, c.W2 = self.W1.copy(), self.W2.copy()
        return c


def q_learning():
    mostrar(21, "Q-Learning", "Aprende Q(s,a) usando o MELHOR valor do próximo estado (off-policy).")
    rng = np.random.default_rng(0)
    Q = np.zeros((5, 2))

    def atualiza(s, a, r, s2, a2, fim):
        alvo = r + 0.9 * Q[s2].max() * (1 - fim)
        Q[s, a] += 0.5 * (alvo - Q[s, a])

    rodar_corredor(Q, atualiza, 200, rng)
    print("Out: política:", Q.argmax(1)[:4].tolist(), "Q(0, dir):", round(float(Q[0, 1]), 3))


def sarsa():
    mostrar(22, "SARSA", "Igual, mas usa a ação que REALMENTE vai tomar (on-policy).")
    rng = np.random.default_rng(0)
    Q = np.zeros((5, 2))

    def atualiza(s, a, r, s2, a2, fim):
        alvo = r + 0.9 * Q[s2, a2] * (1 - fim)
        Q[s, a] += 0.5 * (alvo - Q[s, a])

    rodar_corredor(Q, atualiza, 200, rng)
    print("Out: política:", Q.argmax(1)[:4].tolist(), "Q(0, dir):", round(float(Q[0, 1]), 3))


def dqn_mini():
    mostrar(23, "DQN (mini)", "Troca a tabela Q por uma rede + replay buffer + rede-alvo.")
    rng = np.random.default_rng(0)
    buf = []
    for _ in range(500):
        s = int(rng.integers(4))
        a = int(rng.integers(2))
        s2, r, f = passo(s, a)
        buf.append((s, a, r, s2, f))
    net = MiniRede(rng)
    alvo = net.copia()
    for t in range(4000):
        s, a, r, s2, f = buf[int(rng.integers(len(buf)))]
        net.treina(s, a, r + 0.9 * alvo.q(s2).max() * (1 - f))
        if t % 100 == 0:
            alvo = net.copia()
    pol = [int(net.q(s).argmax()) for s in range(4)]
    print("Out: política:", pol)


def reinforce():
    mostrar(24, "Policy Gradient (REINFORCE)", "Ajusta direto as probabilidades das ações pela recompensa.")
    rng = np.random.default_rng(0)
    PB = np.array([0.2, 0.5, 0.8])
    th = np.zeros(3)
    for _ in range(2000):
        p = softmax(th)
        a = int(rng.choice(3, p=p))
        r = float(rng.random() < PB[a])
        th += 0.1 * r * (np.eye(3)[a] - p)
    print("Out: probabilidades:", np.round(softmax(th), 2).tolist())


def actor_critic():
    mostrar(25, "Actor-Critic", "Ator escolhe ações; crítico estima o valor e reduz a variância.")
    rng = np.random.default_rng(0)
    PB = np.array([0.2, 0.5, 0.8])
    th, V = np.zeros(3), 0.0
    for _ in range(2000):
        p = softmax(th)
        a = int(rng.choice(3, p=p))
        d = float(rng.random() < PB[a]) - V
        th += 0.1 * d * (np.eye(3)[a] - p)
        V += 0.05 * d
    print("Out: probabilidades:", np.round(softmax(th), 2).tolist(), "V:", round(float(V), 2))


def rodar():
    print("\n— Aprendizado por reforço · corredor de 5 casas e bandit de 3 braços —")
    q_learning()
    sarsa()
    dqn_mini()
    reinforce()
    actor_critic()


if __name__ == "__main__":
    rodar()
