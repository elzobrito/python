"""Reavaliação e relatório com bootstrap hierárquico e métricas originais."""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from .agents import load_agent
from .experiment import environment,rollout,write_csv

def interval(matrix,seed=812,alpha=.05,resamples=2000):
    """Reamostra runs e seeds ambientais; preserva pares em diferenças."""
    m=np.asarray(matrix,dtype=float)
    if m.ndim!=2 or not m.size or not np.isfinite(m).all():raise ValueError('Matriz experimental invalida')
    rng=np.random.default_rng(seed)
    runs=rng.integers(m.shape[0],size=(resamples,m.shape[0]))
    episodes=rng.integers(m.shape[1],size=(resamples,m.shape[1]))
    boot=m[runs[:,:,None],episodes[:,None,:]].mean(axis=(1,2))
    return [float(x) for x in np.quantile(boot,[alpha/2,1-alpha/2])]

def matrix(rows,game,candidate,config):
    selected={(int(r['train_seed']),int(r['eval_seed'])):float(r['score']) for r in rows if r['game']==game and r['candidate']==candidate}
    expected=len(config['train_seeds'])*len(config['test_seeds'])
    actual=sum(r['game']==game and r['candidate']==candidate for r in rows)
    if len(selected)!=expected or actual!=expected:raise ValueError('Partidas ausentes ou duplicadas')
    return np.array([[selected[(s,e)] for e in config['test_seeds']] for s in config['train_seeds']])

def latency(record,config,n=1000):
    agent,metadata=load_agent(record);env=environment(metadata['game'],config)
    obs,_=env.reset(424242);observations=[]
    for _ in range(64):
        observations.append(obs.copy());obs,_,t,tr,_=env.step(agent.act(obs))
        if t or tr:obs,_=env.reset(424243)
    for o in observations:agent.act(o)
    times=[]
    for i in range(n):
        o=observations[i%len(observations)];start=time.perf_counter_ns();agent.act(o)
        times.append((time.perf_counter_ns()-start)/1000)
    env.close();return dict(median_us=float(np.median(times)),p95_us=float(np.percentile(times,95)),calls=n)

def read_csv(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))

def build_report(out):
    out=Path(out);config=json.loads((out/'config.json').read_text());models=json.loads((out/'models.json').read_text())
    rows=read_csv(out/'evaluation.csv');validation=read_csv(out/'validation.csv')
    costs=json.loads((out/'costs.json').read_text());summary=[];decisions={}
    from .experiment import CANDIDATES
    enough=len(config['train_seeds'])>=5 and len(config['test_seeds'])>=100
    for game in config['games']:
        matrices={c:matrix(rows,game,c,config) for c in CANDIDATES[game]}
        learned=[c for c in CANDIDATES[game] if c not in ['random','heuristic']]
        winner=max(learned,key=lambda c:np.mean([float(r['score']) for r in validation if r['game']==game and r['candidate']==c]))
        for candidate,m in matrices.items():
            matching=[r for r in rows if r['game']==game and r['candidate']==candidate]
            timing=[latency(models[f'{game}-{candidate}-{s}'],config) for s in config['train_seeds']]
            meta=[c for c in costs if c['game']==game and c['candidate']==candidate]
            diffs={b:dict(mean=float((m-matrices[b]).mean()),ci=interval(m-matrices[b],alpha=.025)) for b in ['random','heuristic']}
            p95=max(t['p95_us'] for t in timing)
            summary.append(dict(game=game,candidate=candidate,n_runs=len(m),n_episodes=m.size,mean=float(m.mean()),median=float(np.median(m)),std=float(m.std(ddof=1)) if m.size>1 else 0.,ci95=interval(m),win_rate=None if game=='invaders' else float(np.mean([r['win']=='True' for r in matching])),truncation_rate=float(np.mean([r['truncated']=='True' for r in matching])),latency_p95_us=p95,latency_median_us=float(np.median([t['median_us'] for t in timing])),train_wall_mean_s=float(np.mean([c['wall_seconds'] for c in meta])),train_cpu_mean_s=float(np.mean([c['cpu_seconds'] for c in meta])),transitions=sum(c['transitions'] for c in meta),comparisons=diffs,selected_by_validation=candidate==winner))
        selected=next(r for r in summary if r['game']==game and r['candidate']==winner)
        supported=enough and selected['latency_p95_us']<=1000 and all(v['ci'][0]>0 for v in selected['comparisons'].values())
        decisions[game]=dict(validation_candidate=winner,status='SUPPORTED_WITHIN_BUDGET' if supported else 'SEM_VENCEDOR_COMPROVADO',recommended=winner if supported else None)
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.suffix in ['.json','.csv'] and p.name not in ['summary.json','artifact_hashes.json']}
    payload=dict(profile=config['profile'],adequate_replications=enough,summary=summary,decisions=decisions)
    (out/'summary.json').write_text(json.dumps(payload,indent=2,allow_nan=False));(out/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2))
    text=['# Benchmark games2d',f"\nPerfil: **{config['profile']}**. {len(config['train_seeds'])} treinamentos por candidato; {len(config['test_seeds'])} partidas por modelo.",
          '\nPontuação original; shaping excluído da avaliação. IC95 de média por bootstrap hierárquico. Diferenças pareadas usam IC97,5 para duas comparações do candidato selecionado exclusivamente pela validação.',
          '\nOrçamentos e horizonte constam em config.json. Tempo de treino não significa convergência. Latência inclui overhead da chamada/timer; p95 reportado é o máximo entre runs.',
          '\n| Jogo | Agente | Média | IC95 | Mediana | Desvio | p95 µs | CPU treino s | Truncamentos |',
          '|---|---|---:|---|---:|---:|---:|---:|---:|']
    for r in summary:text.append(f"| {r['game']} | {r['candidate']} | {r['mean']:.2f} | [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}] | {r['median']:.2f} | {r['std']:.2f} | {r['latency_p95_us']:.1f} | {r['train_cpu_mean_s']:.2f} | {r['truncation_rate']:.1%} |")
    text+=['\n## Decisões']
    for game,d in decisions.items():text.append(f"\n- {game}: **{d['status']}**; candidato escolhido na validação: `{d['validation_candidate']}`.")
    text+=['\n## Limitações','\nObservações Snake/Invaders são parciais. Horizons e física são variantes documentadas. Ruído de coleta não garante recuperação da imitação. Smoke não prova eficácia; pilot usa treino reduzido. Avaliação automatizada não representa aceite humano nem revisão independente.','\n## Artefatos','\nconfig.json, provenance.json, costs.json, models.json, training.csv, validation.csv, evaluation.csv, summary.json, artifact_hashes.json e modelos representativos selecionados pela validação.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
    plot_curves(out)
    return payload

def plot_curves(out):
    path=Path(out)/'training.csv'
    if not path.exists():return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:return
    rows=read_csv(path);groups=sorted({(r['game'],r['candidate']) for r in rows})
    if not groups:return
    fig,axes=plt.subplots(len(groups),1,figsize=(9,2.5*len(groups)),squeeze=False)
    for ax,(game,candidate) in zip(axes[:,0],groups):
        selected=[r for r in rows if (r['game'],r['candidate'])==(game,candidate)]
        for seed in sorted({r['train_seed'] for r in selected}):
            values=[r for r in selected if r['train_seed']==seed]
            x=np.array([float(r['unit']) for r in values]);y=np.array([float(r['score']) for r in values])
            w=50 if len(y)>=50 else 1;smooth=np.convolve(y,np.ones(w)/w,mode='valid')
            ax.plot(x[w-1:],smooth,label=f'seed {seed}',alpha=.8,marker='o' if len(y)<50 else None)
        ax.set_title(f'{game} / {candidate}');ax.set_xlabel('episódio / step / geração (conforme agente)');ax.set_ylabel('score / fitness');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(Path(out)/'curves.svg');plt.close(fig)
    snake=[r for r in rows if r['game']=='snake' and r['candidate'] in ['q_plain','q_shaping']]
    if snake:
        fig,axes=plt.subplots(2,3,figsize=(13,7),squeeze=False)
        for i,candidate in enumerate(['q_plain','q_shaping']):
            for seed in sorted({r['train_seed'] for r in snake}):
                values=[r for r in snake if r['candidate']==candidate and r['train_seed']==seed]
                if not values:continue
                x=np.array([float(r['unit']) for r in values])
                for ax,field in zip(axes[i],['base_return','train_return','length']):
                    y=np.array([float(r.get(field) or 0) for r in values]);w=min(50,len(y))
                    ax.plot(x[w-1:],np.convolve(y,np.ones(w)/w,'valid'),label=f'seed {seed}')
                    ax.set_title(f'{candidate}: {field}');ax.set_xlabel('episódio');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(Path(out)/'snake-learning.svg');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,help='Reavaliar checkpoints congelados em nova pasta; nunca reescreve o experimento')
    a=p.parse_args()
    if a.out:
        if a.out.resolve()==a.run.resolve() or a.out.exists():raise ValueError('Saida de reavaliacao deve ser nova')
        a.out.mkdir(parents=True)
        import shutil
        for name in ['config.json','models.json','costs.json','validation.csv','provenance.json']:
            shutil.copy2(a.run/name,a.out/name)
        config=json.loads((a.run/'config.json').read_text());models=json.loads((a.run/'models.json').read_text());rows=[]
        for record in models.values():
            agent,meta=load_agent(record);env=environment(meta['game'],config)
            for s in config['test_seeds']:
                rows.append(dict(game=meta['game'],candidate=meta['candidate'],train_seed=meta['seed'],eval_seed=s,**rollout(env,agent,s,s+1000000+meta['seed']*10000)))
        write_csv(a.out/'evaluation.csv',rows);build_report(a.out)
    else:
        # Verificação read-only de evidências congeladas; geração só durante run/reavaliação.
        expected=json.loads((a.run/'artifact_hashes.json').read_text())
        for name,digest in expected.items():
            if hashlib.sha256((a.run/name).read_bytes()).hexdigest()!=digest:raise ValueError(f'Hash divergente: {name}')
        print((a.run/'REPORT.md').read_text())

if __name__=='__main__':main()
