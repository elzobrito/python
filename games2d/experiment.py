"""Treino e avaliação sem janela, com budgets e separação de sementes explícitos."""
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
import numpy as np
from .environments import make_env
from .agents import (RandomAgent,Heuristic,QLearning,DQN,ReplayBuffer,CART,LinearAgent,evolve_population,save_agent,load_agent)

CANDIDATES={'snake':['random','heuristic','q_plain','q_shaping','genetic'],
            'pong':['random','heuristic','cart','dqn'],
            'breakout':['random','heuristic','dqn'],
            'invaders':['random','heuristic','genetic']}
PROFILES={
    'smoke':dict(train_seeds=[0],test_count=10,validation_count=3,q_episodes=15,dqn_steps=400,cart_samples=300,ga_population=4,ga_elite=2,ga_generations=2,ga_games=2,max_steps=180),
    'pilot':dict(train_seeds=list(range(5)),test_count=100,validation_count=20,q_episodes=2000,dqn_steps=3000,cart_samples=10000,ga_population=6,ga_elite=2,ga_generations=3,ga_games=3,max_steps=1200),
    'research':dict(train_seeds=list(range(5)),test_count=100,validation_count=20,q_episodes=2000,dqn_steps=20000,cart_samples=10000,ga_population=30,ga_elite=10,ga_generations=20,ga_games=5,max_steps=3600),
}

def environment(game,config,shaping=False):
    limit=min(config['max_steps'],2000 if game=='snake' else 1200 if game=='invaders' else 3600)
    extra={'shaping':shaping} if game=='snake' and shaping else {}
    if game=='snake' and 'snake_bits' in config:extra['bits']=config['snake_bits']
    return make_env(game,max_steps=limit,**extra)

def rollout(env,agent,seed,policy_seed=None):
    obs,info=env.reset(seed)
    if isinstance(agent,RandomAgent):agent.rng=np.random.default_rng(policy_seed if policy_seed is not None else seed+1000000)
    base=shaped=0.
    while True:
        a=agent.act(obs);obs,r,term,trunc,info=env.step(a)
        base+=info['base_reward'];shaped+=r
        if term or trunc:break
    return dict(score=float(info['score']),base_return=base,train_return=shaped,steps=info['steps'],terminated=term,truncated=trunc,reason=info['reason'],win=info['win'],apples=info.get('apples',0),lives=info.get('lives',0),wave=info.get('wave',0))

def epsilon(step,budget):return max(.01,float(.01**(step/max(1,budget-1))))

def train_agent(game,candidate,seed,config):
    rng=np.random.default_rng(np.random.SeedSequence([seed,2718]))
    def episode_seed():return int(rng.integers(10000000,2**31))
    env=environment(game,config,shaping=candidate=='q_shaping')
    trace=[];transitions=0;start=time.perf_counter();cpu=time.process_time()
    if candidate=='random':agent=RandomAgent(env.n_actions,seed)
    elif candidate=='heuristic':agent=Heuristic(game)
    elif candidate in ['q_plain','q_shaping']:
        agent=QLearning(seed=seed)
        for episode in range(config['q_episodes']):
            obs,_=env.reset(episode_seed());base=ret=0.
            while True:
                a=agent.act(obs,epsilon(episode,config['q_episodes']))
                nxt,r,term,trunc,info=env.step(a);agent.learn(obs,a,r,nxt,term)
                obs=nxt;base+=info['base_reward'];ret+=r;transitions+=1
                if term or trunc:break
            trace.append(dict(unit=episode,score=info['score'],base_return=base,train_return=ret,steps=info['steps'],loss=0.,length=info['length']))
    elif candidate=='dqn':
        agent=DQN(env.observation_size,hidden=32 if game=='breakout' else 16,seed=seed)
        buf=ReplayBuffer(10000,env.observation_size);obs,_=env.reset(episode_seed());base=ret=0.;loss=0.
        for step in range(config['dqn_steps']):
            a=agent.act(obs,epsilon(step,config['dqn_steps']))
            nxt,r,term,trunc,info=env.step(a);buf.add(obs,a,r,nxt,term);obs=nxt
            transitions+=1;base+=info['base_reward'];ret+=r
            if buf.size>=256 and (step+1)%4==0:loss=agent.train_batch(buf.sample(32,rng))
            if (step+1)%200==0:agent.sync_target()
            if term or trunc:
                trace.append(dict(unit=step,score=info['score'],base_return=base,train_return=ret,steps=info['steps'],loss=loss))
                obs,_=env.reset(episode_seed());base=ret=0.
        if base or ret or env.steps:trace.append(dict(unit=config['dqn_steps'],score=env.score,base_return=base,train_return=ret,steps=env.steps,loss=loss))
    elif candidate=='cart':
        expert=Heuristic(game);X=[];y=[];obs,_=env.reset(episode_seed())
        for step in range(config['cart_samples']):
            label=expert.act(obs);X.append(obs.copy());y.append(label)
            # Perturbar a execução, manter rótulo correto do estado visitado.
            action=int(rng.integers(env.n_actions)) if rng.random()<.05 else label
            obs,r,term,trunc,info=env.step(action);transitions+=1
            if term or trunc:obs,_=env.reset(episode_seed())
        agent=CART();agent.fit(np.array(X),np.array(y))
        trace.append(dict(unit=len(X),score=0,base_return=0,train_return=0,steps=len(X),loss=0.))
    elif candidate=='genetic':
        bits=config.get('snake_bits',11) if game=='snake' else 6
        shape=(bits,3) if game=='snake' else (6,)
        population=rng.normal(0,1,(config['ga_population'],*shape))
        if game=='invaders':population[0]=[1,5,1,3,2,3]
        best_score=-np.inf;best=None
        # Mesmas sementes dentro de cada geração para todos os indivíduos.
        for generation in range(config['ga_generations']):
            seeds=[episode_seed() for _ in range(config['ga_games'])];fitness=[]
            for weights in population:
                policy=LinearAgent(game,weights);rows=[rollout(env,policy,s) for s in seeds]
                transitions+=sum(r['steps'] for r in rows)
                fitness.append(float(np.mean([100*r['apples']-.001*r['steps'] if game=='snake' else r['score'] for r in rows])))
            # Campeão da geração final; não comparar fitness de conjuntos diferentes.
            index=int(np.argmax(fitness));best=population[index].copy();best_score=fitness[index]
            trace.append(dict(unit=generation,score=best_score,base_return=0,train_return=0,steps=transitions,loss=0.))
            if generation+1<config['ga_generations']:population=evolve_population(population,fitness,rng,elite=config['ga_elite'])
        agent=LinearAgent(game,best)
    else:raise ValueError(candidate)
    metadata=dict(game=game,candidate=candidate,seed=seed,wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,transitions=transitions,updates=getattr(agent,'updates',0))
    env.close();return agent,metadata,trace

def write_csv(path,rows):
    if not rows:raise ValueError('Sem linhas para CSV')
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def source_hashes():
    paths=list(Path(__file__).parent.glob('*.py'))+[Path(__file__).parent.parent/'arvores.py',Path(__file__).parent/'PROTOCOL.md']
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.exists()}

def machine():
    cpu='unknown'
    try:
        cpu=next(line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    except (OSError,StopIteration):pass
    return dict(python=sys.version,numpy=np.__version__,platform=platform.platform(),cpu=cpu)

def run(config,out,games):
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Diretorio de saida deve estar vazio (preserva evidencias)')
    out.mkdir(parents=True,exist_ok=True)
    config=dict(config,games=games,schema='games2d.experiment.v1',validation_seeds=list(range(500,500+config['validation_count'])),test_seeds=list(range(1000,1000+config['test_count'])))
    (out/'config.json').write_text(json.dumps(config,indent=2))
    (out/'provenance.json').write_text(json.dumps(dict(machine=machine(),sources=source_hashes()),indent=2))
    evaluation=[];validation=[];training=[];models={};costs=[]
    for game in games:
        for candidate in CANDIDATES[game]:
            for seed in config['train_seeds']:
                print(f'{game}/{candidate}/seed={seed}',flush=True)
                agent,meta,trace=train_agent(game,candidate,seed,config)
                identifier=f'{game}-{candidate}-{seed}'
                record=dict(schema='games2d.model.v1',metadata=meta,agent=agent.data())
                models[identifier]=record;costs.append(meta)
                for row in trace:training.append(dict(game=game,candidate=candidate,train_seed=seed,**row))
                # Avaliar somente recompensa base: o ambiente de teste nunca aplica shaping.
                for phase,seeds,rows in [('validation',config['validation_seeds'],validation),('test',config['test_seeds'],evaluation)]:
                    env=environment(game,config)
                    for s in seeds:
                        rows.append(dict(game=game,candidate=candidate,train_seed=seed,eval_seed=s,**rollout(env,agent,s,policy_seed=s+1000000+seed*10000)))
                    env.close()
    (out/'models.json').write_text(json.dumps(models,allow_nan=False))
    (out/'costs.json').write_text(json.dumps(costs,indent=2))
    write_csv(out/'evaluation.csv',evaluation);write_csv(out/'validation.csv',validation)
    if training:write_csv(out/'training.csv',training)
    # Modelo representativo escolhido exclusivamente pela validação, por candidato.
    for game in games:
        for candidate in CANDIDATES[game]:
            if candidate in ['random','heuristic']:continue
            scores={s:np.mean([r['score'] for r in validation if r['game']==game and r['candidate']==candidate and r['train_seed']==s]) for s in config['train_seeds']}
            seed=max(scores,key=scores.get)
            record=models[f'{game}-{candidate}-{seed}'];record=dict(record,metadata=dict(record['metadata'],selection='validation_mean',validation_mean=float(scores[seed])))
            (out/f'{game}-{candidate}.json').write_text(json.dumps(record,allow_nan=False))
    from .benchmark import build_report
    build_report(out)
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=PROFILES,default='smoke');p.add_argument('--games',nargs='+',choices=CANDIDATES,default=list(CANDIDATES))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--config',type=Path,help='JSON com overrides explicitos, registrados na saida')
    args=p.parse_args();config=dict(PROFILES[args.profile],profile=args.profile)
    if args.config:
        extra=json.loads(args.config.read_text())
        unknown=set(extra)-set(config)
        if unknown:raise ValueError(f'Chaves desconhecidas: {unknown}')
        config.update(extra)
    for key in ['test_count','validation_count','q_episodes','dqn_steps','cart_samples','ga_population','ga_elite','ga_generations','ga_games','max_steps']:
        if not isinstance(config[key],int) or config[key]<1:raise ValueError(f'Budget invalido: {key}')
    if not config['train_seeds'] or len(set(config['train_seeds']))!=len(config['train_seeds']) or any(not isinstance(s,int) or s<0 for s in config['train_seeds']):raise ValueError('train_seeds invalidas')
    if config['validation_count']>500:raise ValueError('Validacao invade seeds de teste')
    if len(set(args.games))!=len(args.games):raise ValueError('Jogos duplicados')
    if config['ga_elite']>=config['ga_population']:raise ValueError('Elite deve ser menor que populacao')
    print(run(config,args.out,args.games))

if __name__=='__main__':main()
