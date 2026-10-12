"""Políticas e operadores matemáticos em NumPy; modelos de inferência em JSON."""
import json
from pathlib import Path
import numpy as np


def state_index(obs):
    x=np.asarray(obs)
    n=x.shape[0] if x.ndim==1 else 0
    if n not in (11,14) or not np.isin(x,[0,1]).all():raise ValueError('Snake requer 11 ou 14 bits')
    return int(x.astype(np.int64) @ (1 << np.arange(n)))

class RandomAgent:
    kind='random'
    def __init__(self,n_actions,seed=0):self.n_actions=n_actions;self.rng=np.random.default_rng(seed)
    def act(self,obs,epsilon=0):return int(self.rng.integers(self.n_actions))
    def data(self):return {'kind':self.kind,'n_actions':self.n_actions}

class Heuristic:
    kind='heuristic'
    def __init__(self,game):self.game=game
    def act(self,obs,epsilon=0):
        if self.game=='snake':
            d=(3,1,0,2)[int(np.argmax(obs[3:7]))]
            dx,dy=obs[9]-obs[10],obs[7]-obs[8]  # desired direction sign reversed below
            desired=np.array([-dx,-dy])
            dirs=np.array([(1,0),(0,1),(-1,0),(0,-1)])
            scores=[-100 if obs[a] else float(dirs[(d+(0,-1,1)[a])%4] @ desired) for a in range(3)]
            return int(np.argmax(scores))
        if self.game=='pong':
            error=(obs[3]+8/500-obs[0])*500
            return 2 if error>10 else 1 if error < -10 else 0
        if self.game=='breakout':
            error=(obs[1]+7/800-obs[0])*800
            return 2 if error>8 else 1 if error < -8 else 0
        if self.game=='invaders':
            dx,dy=obs[6]*800,obs[7]*600
            if dy<60 and abs(dx)<45:return 1 if dx>=0 else 2
            target=(obs[2]+obs[3])/2
            action=1 if obs[0]>target+.02 else 2 if obs[0]<target-.02 else 0
            return action+3 if obs[1]<=0 else action
        raise ValueError(self.game)
    def data(self):return {'kind':self.kind,'game':self.game}

class QLearning:
    kind='q_learning'
    def __init__(self,seed=0,alpha=.1,gamma=.9,n_states=2048):
        self.Q=np.zeros((n_states,3));self.rng=np.random.default_rng(seed);self.alpha,self.gamma=alpha,gamma
    def _ensure_state(self,idx):
        if idx>=len(self.Q):
            new_size=16384 if idx<16384 else idx+1
            new_Q=np.zeros((new_size,3));new_Q[:len(self.Q)]=self.Q;self.Q=new_Q
    def act(self,obs,epsilon=0):
        idx=state_index(obs);self._ensure_state(idx)
        return int(self.rng.integers(3)) if self.rng.random()<epsilon else int(self.Q[idx].argmax())
    def learn(self,obs,a,r,nxt,terminated):
        s,s2=state_index(obs),state_index(nxt)
        self._ensure_state(max(s,s2))
        target=r+self.gamma*(not terminated)*self.Q[s2].max()
        self.Q[s,a]+=self.alpha*(target-self.Q[s,a])
    def data(self):return {'kind':self.kind,'Q':self.Q.tolist(),'alpha':self.alpha,'gamma':self.gamma}

class ReplayBuffer:
    def __init__(self,capacity,inputs):
        self.capacity=capacity;self.pos=self.size=0
        self.obs=np.zeros((capacity,inputs));self.nxt=np.zeros_like(self.obs)
        self.actions=np.zeros(capacity,dtype=int);self.rewards=np.zeros(capacity);self.terminal=np.zeros(capacity,dtype=bool)
    def add(self,obs,a,r,nxt,terminated):
        i=self.pos;self.obs[i]=obs;self.nxt[i]=nxt;self.actions[i]=a;self.rewards[i]=r;self.terminal[i]=terminated
        self.pos=(i+1)%self.capacity;self.size=min(self.size+1,self.capacity)
    def sample(self,n,rng):
        if self.size<n:raise ValueError('Replay ainda sem minibatch')
        idx=rng.choice(self.size,n,replace=False)
        return self.obs[idx],self.actions[idx],self.rewards[idx],self.nxt[idx],self.terminal[idx]

class DQN:
    kind='dqn'
    def __init__(self,inputs,hidden=16,actions=3,seed=0,lr=.001,gamma=.99):
        self.inputs,self.hidden,self.actions=inputs,hidden,actions
        self.rng=np.random.default_rng(seed);self.lr,self.gamma=lr,gamma
        self.W1=self.rng.normal(0,1/np.sqrt(inputs),(inputs,hidden));self.b1=np.zeros(hidden)
        self.W2=self.rng.normal(0,1/np.sqrt(hidden),(hidden,actions));self.b2=np.zeros(actions)
        self.updates=0;self.sync_target()
    def sync_target(self):self.target=[p.copy() for p in (self.W1,self.b1,self.W2,self.b2)]
    def q(self,obs,target=False):
        x=np.asarray(obs,dtype=float)
        if x.shape[-1]!=self.inputs or not np.isfinite(x).all():raise ValueError('Observacao invalida para DQN')
        w,b,v,c=self.target if target else (self.W1,self.b1,self.W2,self.b2)
        return np.tanh(x@w+b)@v+c
    def act(self,obs,epsilon=0):
        return int(self.rng.integers(self.actions)) if self.rng.random()<epsilon else int(self.q(obs).argmax())
    def train_batch(self,batch):
        x,a,r,nxt,terminal=batch;n=len(x)
        hidden=np.tanh(x@self.W1+self.b1);q=hidden@self.W2+self.b2
        targets=r+self.gamma*(1-terminal)*self.q(nxt,target=True).max(axis=1)
        err=q[np.arange(n),a]-targets
        loss=np.where(abs(err)<=1,.5*err**2,abs(err)-.5).mean()
        delta=np.zeros_like(q);delta[np.arange(n),a]=np.clip(err,-1,1)/n
        dh=(delta@self.W2.T)*(1-hidden**2)
        grads=[x.T@dh,dh.sum(axis=0),hidden.T@delta,delta.sum(axis=0)]
        norm=np.sqrt(sum(float((g*g).sum()) for g in grads));scale=min(1.,10/max(norm,1e-12))
        for p,g in zip((self.W1,self.b1,self.W2,self.b2),grads):p-=self.lr*scale*g
        if not all(np.isfinite(p).all() for p in (self.W1,self.b1,self.W2,self.b2)):raise FloatingPointError('DQN divergiu')
        self.updates+=1;return float(loss)
    def data(self):
        return dict(kind=self.kind,inputs=self.inputs,hidden=self.hidden,actions=self.actions,lr=self.lr,gamma=self.gamma,updates=self.updates,**{k:getattr(self,k).tolist() for k in ['W1','b1','W2','b2']})

class CART:
    kind='cart'
    def __init__(self,tree=None):self.tree=tree
    def fit(self,X,y):
        from arvores import crescer_class
        root=crescer_class(np.asarray(X),np.asarray(y,dtype=int),0,6,3,min_folha=8)
        def encode(node):
            return {'value':int(node.valor)} if node.folha else {'feature':int(node.j),'threshold':float(node.t),'left':encode(node.esq),'right':encode(node.dir)}
        self.tree=encode(root)
    def act(self,obs,epsilon=0):
        if self.tree is None:raise RuntimeError('CART nao treinada')
        node=self.tree
        while 'feature' in node:node=node['left'] if obs[node['feature']]<=node['threshold'] else node['right']
        return int(node['value'])
    def data(self):return {'kind':self.kind,'tree':self.tree}

def utility_features(obs):
    # Predição de deslocamento torna as features específicas de cada ação.
    movement=np.array([0,-1,1,0,-1,1])*6/800
    firing=np.array([0,0,0,1,1,1],dtype=float)
    nxt=np.clip(obs[0]+movement,25/800,775/800)
    align=1-abs(nxt-(obs[2]+obs[3])/2)
    threat=obs[7]<.12 and abs(obs[6])<.12
    safety=np.minimum(abs(obs[6]-movement)*20,1) if threat else np.zeros(6)
    ready=firing*float(obs[1]<=0)
    shield=-firing*obs[8]
    boundary=-abs(obs[0]+movement-nxt)*100
    aim=ready*align
    return np.column_stack([align,safety,ready,shield,boundary,aim])

class LinearAgent:
    kind='linear'
    def __init__(self,game,weights):
        self.game=game;self.weights=np.asarray(weights,dtype=float)
        expected=(14,3) if (game=='snake' and self.weights.shape==(14,3)) else ((11,3) if game=='snake' else (6,))
        if self.weights.shape!=expected or not np.isfinite(self.weights).all():raise ValueError('Pesos lineares invalidos')
    def act(self,obs,epsilon=0):
        values=obs@self.weights if self.game=='snake' else utility_features(obs)@self.weights
        return int(values.argmax())
    def data(self):return {'kind':self.kind,'game':self.game,'weights':self.weights.tolist()}

def save_agent(agent,path,metadata):
    data={'schema':'games2d.model.v1','metadata':metadata,'agent':agent.data()}
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding='utf-8')

def load_agent(path,expected_game=None):
    data=path if isinstance(path,dict) else json.loads(Path(path).read_text())
    if data.get('schema')!='games2d.model.v1':raise ValueError('Schema do modelo desconhecido')
    game=data['metadata']['game']
    if expected_game is not None and game!=expected_game:raise ValueError('Modelo pertence a outro jogo')
    d=data['agent'];kind=d['kind']
    if kind=='random':agent=RandomAgent(d['n_actions'],data['metadata'].get('seed',0))
    elif kind=='heuristic':agent=Heuristic(d['game'])
    elif kind=='q_learning':
        agent=QLearning(alpha=d['alpha'],gamma=d['gamma']);agent.Q=np.asarray(d['Q'],dtype=float)
        if agent.Q.shape not in ((2048,3),(16384,3)) or not np.isfinite(agent.Q).all():raise ValueError('Tabela Q invalida')
    elif kind=='dqn':
        agent=DQN(d['inputs'],d['hidden'],d['actions'],lr=d['lr'],gamma=d['gamma'])
        for k in ['W1','b1','W2','b2']:
            x=np.asarray(d[k],dtype=float)
            if x.shape!=getattr(agent,k).shape or not np.isfinite(x).all():raise ValueError('Pesos DQN invalidos')
            setattr(agent,k,x)
        agent.updates=d['updates'];agent.sync_target()
    elif kind=='cart':agent=CART(d['tree'])
    elif kind=='linear':agent=LinearAgent(d['game'],d['weights'])
    else:raise ValueError('Tipo de agente desconhecido')
    return agent,data['metadata']

def evolve_population(population,fitness,rng,elite=10,mutation=.1):
    """Seleção elitista, crossover uniforme e mutação gaussiana (pesos reais)."""
    p=np.asarray(population,dtype=float);f=np.asarray(fitness,dtype=float)
    if len(p)<2 or f.shape!=(len(p),) or not np.isfinite(f).all():raise ValueError('Populacao/fitness invalidos')
    if not 1<=elite<len(p):raise ValueError('elite invalida')
    parents=p[np.argsort(f)[-elite:]].copy()
    shape=(len(p)-elite,*p.shape[1:]);a=parents[rng.integers(elite,size=shape[0])];b=parents[rng.integers(elite,size=shape[0])]
    children=np.where(rng.random(shape)<.5,a,b)
    children+=rng.normal(0,mutation,shape)*(rng.random(shape)<.1)
    return np.concatenate([parents,children])
