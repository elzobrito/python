import sys
import numpy as np
import pytest
from games2d import make_env
from games2d.env_base import Rect

@pytest.mark.parametrize('game',['snake','pong','breakout','invaders'])
def test_reproducible_headless(game):
    a,b=make_env(game,max_steps=40),make_env(game,max_steps=40)
    x,_=a.reset(17);y,_=b.reset(17);np.testing.assert_array_equal(x,y)
    rng=np.random.default_rng(42)
    for _ in range(40):
        action=int(rng.integers(a.n_actions))
        ra,rb=a.step(action),b.step(action)
        np.testing.assert_array_equal(ra[0],rb[0]);assert ra[1:]==rb[1:]
        assert np.isfinite(ra[0]).all()
        if ra[2] or ra[3]:break
    with pytest.raises(RuntimeError):a.step(0)
    assert 'pygame' not in sys.modules

@pytest.mark.parametrize('game',['snake','pong','breakout','invaders'])
def test_action_and_observation_contract(game):
    env=make_env(game);obs,_=env.reset(1);assert obs.shape==(env.observation_size,)
    with pytest.raises(ValueError):env.step(env.n_actions)
    obs[:]=999;assert not (env.observe()==999).any()

def test_snake_tail_and_food_and_death():
    e=make_env('snake');e.reset(0)
    assert e.food not in e.body
    e.body=[(1,1),(1,2),(0,2),(0,1)];e.direction=2;e.food=(4,4)
    assert not e.danger(0)
    _,_,term,_,_=e.step(0);assert not term
    e.body=[(0,1),(1,1),(1,2)];e.direction=2
    _,r,term,trunc,info=e.step(0);assert term and not trunc and r==-10 and info['reason']=='collision'

def test_snake_growth_and_truncation():
    e=make_env('snake',starvation=1);e.reset(0);e.food=(21,15)
    _,r,t,tr,info=e.step(0);assert r==10 and len(e.body)==4 and not t and not tr
    e.food=(0,0)
    _,r,t,tr,info=e.step(0);assert tr and not t and r==0 and info['reason']=='starvation_limit'

def test_shaping_telescopes_and_terminal_zero():
    e=make_env('snake',shaping=True,max_steps=5);e.reset(2);initial=e.potential();total=0.
    for t in range(5):
        _,r,term,trunc,info=e.step(0);total+=e.gamma**t*info['shaping_reward']
        if term or trunc:break
    final=0 if term else e.potential()
    assert total==pytest.approx(-initial+e.gamma**(t+1)*final)
    e.reset(2);e.body=[(0,0)];e.direction=2;phi=e.potential()
    _,r,t,tr,i=e.step(0);assert t and i['shaping_reward']==-phi and i['base_reward']==-10

def test_pong_point_and_speed_limit():
    e=make_env('pong');e.reset(0);e.vx=-799;e.ball=Rect(40,210,16,16)
    e.step(0);assert 0<e.vx<=800
    e.points=[4,0];e.ball.x=801;e.vx=300
    _,r,t,tr,info=e.step(0);assert r==1 and t and info['win']

def test_breakout_resistance_and_lives():
    e=make_env('breakout');e.reset(1);e.ball=Rect(20,80,14,14);e.vx=0;e.vy=-340
    e.step(0);assert e.hp[0]==1 and e.score==10
    e.lives=1;e.ball.y=610
    _,r,t,_,i=e.step(0);assert t and i['lives']==0 and r==-2

def test_invaders_combined_fire_and_shield():
    e=make_env('invaders');e.reset(0);x=e.ship.x
    e.step(4);assert e.ship.x<x and len(e.shots)==1
    e.shots=[Rect(110,465,4,12)];before=len(e.shields)
    e.step(0);assert len(e.shields)==before-1 and not e.shots

def test_invaders_wave_and_base():
    e=make_env('invaders');e.reset(0);e.aliens=np.empty((0,2));e.step(0);assert e.wave==2 and len(e.aliens)==50
    e.aliens[:,1]=550
    _,r,t,_,i=e.step(0);assert t and i['reason']=='base' and r==-10

from games2d.agents import QLearning, DQN, ReplayBuffer, Heuristic, LinearAgent, CART, save_agent, load_agent, utility_features

def test_q_bootstrap_terminal_vs_truncation():
    q=QLearning(alpha=1,gamma=.9);obs=np.zeros(11);nxt=obs.copy();nxt[0]=1;q.Q[1]=10
    q.learn(obs,0,1,nxt,False);assert q.Q[0,0]==10
    q.learn(obs,1,1,nxt,True);assert q.Q[0,1]==1

def test_dqn_learns_finite_and_frozen_target():
    d=DQN(6,seed=2);x=np.zeros((32,6));a=np.zeros(32,dtype=int)
    target=d.q(x,target=True).copy();before=d.q(x)[0,0]
    for _ in range(100):d.train_batch((x,a,np.ones(32),x,np.ones(32,dtype=bool)))
    assert d.q(x)[0,0]>before
    np.testing.assert_array_equal(d.q(x,target=True),target)
    d.sync_target();np.testing.assert_array_equal(d.q(x),d.q(x,target=True))

def test_replay_wraps():
    b=ReplayBuffer(3,6)
    for i in range(5):b.add(np.ones(6)*i,0,i,np.zeros(6),False)
    assert b.size==3 and set(b.rewards)=={2,3,4}

def test_heuristics_and_utility_depend_on_action():
    for game in ['snake','pong','breakout','invaders']:
        e=make_env(game);o,_=e.reset(4);assert 0<=Heuristic(game).act(o)<e.n_actions
    o=np.array([.5,0,.1,.9,.5,.2,.01,.03,.2]);f=utility_features(o)
    assert f.shape==(6,6) and not np.array_equal(f[1],f[2])

def test_models_roundtrip_and_game_guard(tmp_path):
    models=[QLearning(),DQN(6),LinearAgent('snake',np.ones((11,3))),CART({'value':2})]
    for i,a in enumerate(models):
        game='snake' if a.kind in ['q_learning','linear'] else 'pong'
        o,_=make_env(game).reset(9);p=tmp_path/f'{i}.json'
        save_agent(a,p,{'game':game});b,_=load_agent(p,game)
        assert a.act(o)==b.act(o)
        with pytest.raises(ValueError):load_agent(p,'breakout')

def test_cart_learns_multiclass():
    x=np.column_stack([np.arange(120)/120,np.zeros((120,5))]);y=np.repeat([0,1,2],40)
    a=CART();a.fit(x,y);assert np.mean([a.act(o)==v for o,v in zip(x,y)])>.95

def test_evolution_preserves_elite():
    from games2d.agents import evolve_population
    p=np.arange(24).reshape(4,6).astype(float)
    nxt=evolve_population(p,np.arange(4),np.random.default_rng(0),elite=2)
    assert nxt.shape==p.shape
    np.testing.assert_array_equal(nxt[:2],p[-2:])

from games2d.experiment import train_agent, PROFILES, rollout
from games2d.benchmark import interval, matrix

def test_training_modes_smoke(tmp_path):
    c=dict(PROFILES['smoke'],q_episodes=3,cart_samples=60,dqn_steps=270,ga_generations=1,max_steps=20)
    for game,candidate in [('snake','q_plain'),('snake','q_shaping'),('snake','genetic'),('pong','cart'),('pong','dqn'),('breakout','dqn'),('invaders','genetic')]:
        a,meta,trace=train_agent(game,candidate,0,c)
        assert meta['transitions']>0 and trace
        row=rollout(make_env(game,max_steps=20),a,1000)
        assert row['steps']<=20

def test_statistics_pairs_and_missing_data():
    assert interval(np.ones((5,100))*7)==[7.,7.]
    config={'train_seeds':[0],'test_seeds':[1000,1001]}
    with pytest.raises(ValueError):matrix([], 'snake','random',config)
    with pytest.raises(ValueError):matrix([{'game':'snake','candidate':'random','train_seed':0,'eval_seed':1000,'score':0}]*2,'snake','random',config)

def test_renderer_controls_and_pixels(monkeypatch):
    monkeypatch.setenv('SDL_VIDEODRIVER','dummy');monkeypatch.setenv('SDL_AUDIODRIVER','dummy')
    pg=pytest.importorskip('pygame')
    from games2d.play import Session
    for game in ['snake','pong','breakout','invaders']:
        e=make_env(game);session=Session(e,Heuristic(game),0)
        session.event(pg.event.Event(pg.KEYDOWN,key=pg.K_h));assert session.mode=='humano'
        session.event(pg.event.Event(pg.KEYDOWN,key=pg.K_a));assert session.mode=='IA'
        session.advance();session.draw()
        pixels=pg.surfarray.array3d(session.renderer.screen)
        assert pixels.shape[:2]==(e.width,e.height) and np.unique(pixels.reshape(-1,3),axis=0).shape[0]>10
        session.event(pg.event.Event(pg.KEYDOWN,key=pg.K_r));assert e.steps==0
        session.event(pg.event.Event(pg.KEYDOWN,key=pg.K_ESCAPE));assert not session.running
        session.close()


def test_invaders_heuristic_opens_shield_but_dodges():
    h=Heuristic('invaders')
    obs=np.array([.5,0,.1,.9,.4,.2,1.,1.,1.])
    assert h.act(obs)==3
    obs[6:8]=[.01,.02]
    assert h.act(obs)==1


def test_snake_14_bits_observation_and_qlearning(tmp_path):
    from games2d.agents import QLearning, state_index, save_agent, load_agent
    e = make_env('snake', bits=14)
    obs, info = e.reset(42)
    assert e.observation_size == 14 and obs.shape == (14,)
    # Testar danger2 em proximidade de borda (parede a 2 blocos)
    e.body = [(1, 15), (2, 15), (3, 15)]
    e.direction = 2  # apontando oeste (x=1); 1 passo -> x=0 (livre), 2 passos -> x=-1 (perigo)
    assert not e.danger(0)  # frente a 1 passo livre
    assert e.danger2(0)     # frente a 2 passos bate na parede oeste
    # Testar state_index e QLearning
    idx = state_index(e.observe())
    assert 0 <= idx < 16384
    agent = QLearning(seed=0)
    a = agent.act(e.observe())
    assert a in (0, 1, 2)
    agent.learn(e.observe(), a, 10.0, e.observe(), False)
    assert agent.Q.shape == (16384, 3)
    # Testar serializacao e recarga do modelo de 14 bits
    path = tmp_path / 'snake_14.json'
    save_agent(agent, path, metadata=dict(game='snake', candidate='q_plain', bits=14))
    loaded, meta = load_agent(path, expected_game='snake')
    assert loaded.Q.shape == (16384, 3) and meta['bits'] == 14

