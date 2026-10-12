"""Contrato comum sem dependência de Pygame."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class Rect:
    x: float
    y: float
    w: float
    h: float
    @property
    def right(self): return self.x + self.w
    @property
    def bottom(self): return self.y + self.h
    @property
    def cx(self): return self.x + self.w / 2
    @property
    def cy(self): return self.y + self.h / 2
    def hits(self, other):
        return self.x < other.right and self.right > other.x and self.y < other.bottom and self.bottom > other.y
    def clamp(self, width, height):
        self.x = float(np.clip(self.x, 0, width-self.w))
        self.y = float(np.clip(self.y, 0, height-self.h))

class GameEnvironment(ABC):
    dt = 1/60
    def __init__(self, max_steps):
        if max_steps < 1: raise ValueError('max_steps deve ser positivo')
        self.max_steps = int(max_steps)
        self._renderer = None
        self.ended = True
    def _reset(self, seed):
        self.rng = np.random.default_rng(seed)
        self.steps, self.score, self.ended = 0, 0, False
        self.reason = ''
    def _begin(self, action):
        if self.ended: raise RuntimeError('Chame reset antes de step')
        if not isinstance(action, (int, np.integer)) or isinstance(action, bool) or not 0 <= action < self.n_actions:
            raise ValueError(f'Acao invalida: {action!r}')
        self.steps += 1
    def _finish(self, reward, terminated=False, truncated=False, reason=''):
        if not terminated and not truncated and self.steps >= self.max_steps:
            truncated, reason = True, 'time_limit'
        self.ended = bool(terminated or truncated)
        self.reason = reason if self.ended else ''
        info = self.info()
        info.update(base_reward=float(reward), shaping_reward=0.0, reason=self.reason)
        return self.observe().copy(), float(reward), bool(terminated), bool(truncated), info
    def info(self):
        return {'score': self.score, 'steps': self.steps, 'win': self.reason == 'win'}
    @abstractmethod
    def reset(self, seed=None): pass
    @abstractmethod
    def observe(self): pass
    @abstractmethod
    def step(self, action): pass
    def render(self):
        if self._renderer is None:
            from .play import Renderer
            self._renderer = Renderer(self)
        self._renderer.draw(self)
    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
