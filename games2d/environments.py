"""Núcleos de física compartilhados por treino e renderer; regras em PROTOCOL.md."""
import numpy as np
from .env_base import GameEnvironment, Rect

class Snake(GameEnvironment):
    name, width, height, n_actions, observation_size = 'snake', 800, 600, 3, 11
    cell, dt = 20, 1/8
    directions = ((1,0),(0,1),(-1,0),(0,-1))  # leste sul oeste norte
    def __init__(self, max_steps=2000, starvation=100, shaping=False, gamma=.9, bits=11):
        super().__init__(max_steps)
        if starvation < 1: raise ValueError('starvation deve ser positivo')
        if not 0 <= gamma < 1: raise ValueError('gamma fora de [0,1)')
        if bits not in (11, 14): raise ValueError('bits deve ser 11 ou 14')
        self.starvation, self.shaping, self.gamma, self.bits = starvation, shaping, gamma, bits
        self.observation_size = bits
    def reset(self, seed=None):
        self._reset(seed)
        self.body, self.direction, self.hungry, self.apples = [(20,15),(19,15),(18,15)], 0, 0, 0
        self.food = self._food()
        return self.observe(), self.info()
    def _food(self):
        occupied = set(self.body)
        free = [(x,y) for y in range(30) for x in range(40) if (x,y) not in occupied]
        return free[int(self.rng.integers(len(free)))] if free else None
    def potential(self):
        if self.food is None: return 0.
        x,y = self.body[0];fx,fy = self.food
        return -float(abs(x-fx)+abs(y-fy))
    def next_head(self, action):
        d = (self.direction + (0,-1,1)[action]) % 4
        dx,dy = self.directions[d];x,y = self.body[0]
        return (x+dx,y+dy),d
    def danger(self, action):
        head,_ = self.next_head(action)
        growing = head == self.food
        return not (0 <= head[0] < 40 and 0 <= head[1] < 30) or head in (self.body if growing else self.body[:-1])
    def danger2(self, action):
        """Perigo a 2 passos na direção da ação (olhar 2 blocos à frente)."""
        (x1, y1), d = self.next_head(action)
        dx, dy = self.directions[d]
        x2, y2 = x1 + dx, y1 + dy
        if self.danger(action): return True
        growing = (x1, y1) == self.food
        body_to_check = self.body if growing else (self.body[:-2] if len(self.body) >= 2 else [])
        return not (0 <= x2 < 40 and 0 <= y2 < 30) or (x2, y2) in body_to_check
    def observe(self):
        x,y = self.body[0];fx,fy = self.food if self.food is not None else (x,y)
        d1 = [self.danger(a) for a in range(3)]
        d2 = [self.danger2(a) for a in range(3)] if self.bits == 14 else []
        dir_bits = [self.direction==d for d in (3,1,0,2)]
        food_bits = [fy<y, fy>y, fx<x, fx>x]
        return np.array(d1 + d2 + dir_bits + food_bits, dtype=np.float64)
    def info(self):
        return dict(super().info(), apples=self.apples, length=len(self.body), hungry=self.hungry)
    def step(self, action):
        self._begin(action)
        before = self.potential()
        head,d = self.next_head(action)
        terminated, reward, reason = self.danger(action), 0., ''
        if terminated:
            reward,reason = -10.,'collision'
        else:
            growing = head == self.food
            self.body.insert(0,head);self.direction=d;self.hungry+=1
            if growing:
                self.apples+=1;self.score+=10;self.hungry=0;reward=10.
                self.food=self._food()
                if self.food is None: terminated,reason=True,'win'
            else: self.body.pop()
        truncated = not terminated and self.hungry >= self.starvation
        if truncated: reason='starvation_limit'
        result=self._finish(reward,terminated,truncated,reason)
        shaping=self.gamma*(0. if terminated else self.potential())-before if self.shaping else 0.
        obs,r,term,trunc,info=result;info['shaping_reward']=shaping
        return obs,r+shaping,term,trunc,info

class Pong(GameEnvironment):
    name,width,height,n_actions,observation_size='pong',800,500,3,6
    velocity_max=800.
    def __init__(self,max_steps=3600): super().__init__(max_steps)
    def reset(self,seed=None):
        self._reset(seed)
        self.player,self.opponent=Rect(30,205,12,90),Rect(758,205,12,90)
        self.points=[0,0];self.hits=0;self._serve(1)
        return self.observe(),self.info()
    def _serve(self,direction):
        self.ball=Rect(392,242,16,16);self.vx=300.*direction;self.vy=300.*self.rng.uniform(-.7,.7)
    def observe(self):
        return np.array([self.player.cy/500,self.opponent.cy/500,self.ball.x/800,self.ball.y/500,self.vx/800,self.vy/800],dtype=float)
    def info(self):return dict(super().info(),player_points=self.points[0],opponent_points=self.points[1],hits=self.hits)
    def step(self,action):
        self._begin(action)
        self.player.y+=(0,-1,1)[action]*380*self.dt;self.player.clamp(800,500)
        error=self.ball.cy-self.opponent.cy
        self.opponent.y+=(1 if error>10 else -1 if error < -10 else 0)*285*self.dt
        self.opponent.clamp(800,500)
        self.ball.x+=self.vx*self.dt;self.ball.y+=self.vy*self.dt
        if self.ball.y<=0 or self.ball.bottom>=500:
            self.vy=-self.vy;self.ball.y=float(np.clip(self.ball.y,0,484))
        if self.vx<0 and self.ball.hits(self.player):
            self.vx=min(-self.vx*1.05,800);self.ball.x=self.player.right;self.hits+=1
        elif self.vx>0 and self.ball.hits(self.opponent):
            self.vx=-min(self.vx*1.05,800);self.ball.x=self.opponent.x-self.ball.w
        reward=0.
        if self.ball.right<0:self.points[1]+=1;reward=-1.;self._serve(1)
        elif self.ball.x>800:self.points[0]+=1;reward=1.;self._serve(-1)
        self.score=self.points[0]-self.points[1]
        term=max(self.points)>=5
        return self._finish(reward,term,reason=('win' if self.points[0]>=5 else 'loss') if term else '')

class Breakout(GameEnvironment):
    name,width,height,n_actions,observation_size='breakout',800,600,3,56
    def __init__(self,max_steps=3600):super().__init__(max_steps)
    def reset(self,seed=None):
        self._reset(seed)
        self.paddle=Rect(340,560,120,14);self.lives=3
        self.bricks=[Rect(13+c*78,60+r*28,72,22) for r in range(5) for c in range(10)]
        self.hp=np.array([2 if i<10 else 1 for i in range(50)],dtype=int)
        self._launch();return self.observe(),self.info()
    def _launch(self):self.ball=Rect(self.paddle.cx-7,self.paddle.y-20,14,14);self.vx,self.vy=204.,-340.
    def observe(self):
        return np.r_[self.paddle.cx/800,self.ball.x/800,self.ball.y/600,self.vx/612,self.vy/612,self.lives/3,self.hp/2].astype(float)
    def info(self):return dict(super().info(),lives=self.lives,bricks=int(np.count_nonzero(self.hp)))
    def step(self,action):
        self._begin(action)
        self.paddle.x+=(0,-1,1)[action]*520*self.dt;self.paddle.clamp(800,600)
        self.ball.x+=self.vx*self.dt;self.ball.y+=self.vy*self.dt
        if self.ball.x<=0 or self.ball.right>=800:self.vx=-self.vx;self.ball.x=float(np.clip(self.ball.x,0,786))
        if self.ball.y<=0:self.vy=abs(self.vy);self.ball.y=0
        if self.vy>0 and self.ball.hits(self.paddle):
            self.ball.y=self.paddle.y-self.ball.h
            desvio=float(np.clip((self.ball.cx-self.paddle.cx)/(self.paddle.w/2),-1,1))
            speed=min(np.hypot(self.vx,self.vy)*1.02,612)
            self.vx=speed*.85*desvio;self.vy=-np.sqrt(max(0,speed**2-self.vx**2))
        reward,term,reason=0.,False,''
        if self.ball.y>600:
            self.lives-=1;reward=-2.
            term=self.lives==0;reason='loss' if term else '';self._launch()
        else:
            for i in np.flatnonzero(self.hp):
                b=self.bricks[i]
                if not self.ball.hits(b):continue
                sx=min(self.ball.right-b.x,b.right-self.ball.x)
                sy=min(self.ball.bottom-b.y,b.bottom-self.ball.y)
                if sx<sy:self.vx=-self.vx;self.ball.x+=-sx if self.ball.cx<b.cx else sx
                else:self.vy=-self.vy;self.ball.y+=-sy if self.ball.cy<b.cy else sy
                self.hp[i]-=1;self.score+=10;reward+=1.
                if self.hp[i]==0:self.score+=10
                break
        if not self.hp.any():term,reason=True,'win';reward+=10.
        return self._finish(reward,term,reason=reason)

class Invaders(GameEnvironment):
    name,width,height,n_actions,observation_size='invaders',800,600,6,9
    def __init__(self,max_steps=1200):super().__init__(max_steps)
    def reset(self,seed=None):
        self._reset(seed)
        self.ship=Rect(375,550,50,20);self.cooldown=0.;self.enemy_cooldown=1.
        self.lives,self.wave=3,1;self.shots=[];self.enemy_shots=[]
        self.shields=np.array([(110+e*170+c*10,460+r*10) for e in range(4) for r in range(3) for c in range(8)],dtype=float)
        self._formation();return self.observe(),self.info()
    def _formation(self):
        self.aliens=np.array([(90+c*60,70+r*45) for r in range(5) for c in range(10)],dtype=float)
        self.dx=60+25*(self.wave-1);self.x_accum=0.
    def threat(self):
        if not self.enemy_shots:return None
        candidates=[s for s in self.enemy_shots if s.bottom<=self.ship.bottom]
        if not candidates:return None
        return min(candidates,key=lambda s:(abs(s.cx-self.ship.cx)>self.ship.w/2+12,(self.ship.y-s.bottom)/300))
    def observe(self):
        a=self.aliens
        bounds=(a[:,0].min()/800,(a[:,0].max()+36)/800,(a[:,1].max()+24)/600,self.dx/400) if len(a) else (0,0,0,0)
        t=self.threat();danger=((t.cx-self.ship.cx)/800,(self.ship.y-t.bottom)/600) if t else (1.,1.)
        # Fração de blocos restantes na coluna vertical da nave; tiros próprios também bloqueados.
        above=np.abs(self.shields[:,0]+5-self.ship.cx)<30 if len(self.shields) else np.zeros(0,dtype=bool)
        density=min(float(above.sum())/18,1.)
        return np.array([self.ship.cx/800,max(0,self.cooldown)/.35,*bounds,*danger,density],dtype=float)
    def info(self):return dict(super().info(),lives=self.lives,wave=self.wave,aliens=len(self.aliens))
    @staticmethod
    def _hit_array(shot,positions,w,h):
        if not len(positions):return -1
        hit=np.flatnonzero((shot.x<positions[:,0]+w)&(shot.right>positions[:,0])&(shot.y<positions[:,1]+h)&(shot.bottom>positions[:,1]))
        return int(hit[0]) if len(hit) else -1
    def step(self,action):
        self._begin(action)
        self.ship.x+=(0,-1,1,0,-1,1)[action]*360*self.dt;self.ship.clamp(800,600)
        self.cooldown-=self.dt
        if action>=3 and self.cooldown<=0:
            self.shots.append(Rect(self.ship.cx-2,self.ship.y-12,4,12));self.cooldown=.35
        for s in self.shots:s.y-=520*self.dt
        self.x_accum+=self.dx*self.dt;delta=int(self.x_accum);self.x_accum-=delta
        self.aliens[:,0]+=delta
        if len(self.aliens) and ((self.aliens[:,0]<=10).any() or (self.aliens[:,0]+36>=790).any()):
            self.dx=-float(np.clip(self.dx*1.04,-400,400));self.aliens[:,1]=np.minimum(self.aliens[:,1]+18,600)
            self.aliens[:,0]+=4 if self.dx>0 else -4
        reward=0.;remaining=[]
        for s in self.shots:
            i=self._hit_array(s,self.aliens,36,24)
            if i>=0:self.aliens=np.delete(self.aliens,i,axis=0);self.score+=10;reward+=1.
            else:remaining.append(s)
        self.shots=remaining
        self.enemy_cooldown-=self.dt
        if self.enemy_cooldown<=0 and len(self.aliens):
            x,y=self.aliens[int(self.rng.integers(len(self.aliens)))]
            self.enemy_shots.append(Rect(x+16,y+24,4,12));self.enemy_cooldown=float(self.rng.uniform(.5,1.3))
        for s in self.enemy_shots:s.y+=300*self.dt
        for attr in ('shots','enemy_shots'):
            remaining=[]
            for s in getattr(self,attr):
                i=self._hit_array(s,self.shields,10,10)
                if i>=0:self.shields=np.delete(self.shields,i,axis=0)
                else:remaining.append(s)
            setattr(self,attr,remaining)
        remaining=[]
        for s in self.enemy_shots:
            if s.hits(self.ship):self.lives-=1;reward-=5.
            elif s.y<600:remaining.append(s)
        self.enemy_shots=remaining;self.shots=[s for s in self.shots if s.bottom>0]
        base=bool(len(self.aliens) and (self.aliens[:,1]+24>=self.ship.y).any())
        term=self.lives<=0 or base
        if base:reward-=10.
        if not term and not len(self.aliens):self.wave+=1;self._formation();self.enemy_shots=[]
        return self._finish(reward,term,reason=('base' if base else 'loss') if term else '')

ENVIRONMENTS={'snake':Snake,'pong':Pong,'breakout':Breakout,'invaders':Invaders}
def make_env(game,**kwargs):
    try:cls=ENVIRONMENTS[game]
    except KeyError:raise ValueError(f'Jogo desconhecido: {game}') from None
    return cls(**kwargs)
