"""Menu e demonstração Pygame do mesmo núcleo usado no treinamento."""
import argparse
from pathlib import Path
import numpy as np
from .environments import make_env
from .agents import Heuristic,QLearning,DQN,load_agent

class Renderer:
    def __init__(self,env):
        try:import pygame
        except ImportError:raise RuntimeError('Instale pygame-ce para a demonstracao visual') from None
        self.pg=pygame;pygame.init()
        self.screen=pygame.display.set_mode((env.width,env.height))
        pygame.display.set_caption(f'games2d — {env.name}')
        self.font=pygame.font.Font(None,25);self.big=pygame.font.Font(None,44)
    def rect(self,r,color):
        self.pg.draw.rect(self.screen,color,(round(r.x),round(r.y),round(r.w),round(r.h)))
    def draw(self,env,agent=None,mode='IA',action=None,inference_us=None):
        pg=self.pg;self.screen.fill((15,15,25))
        if env.name=='snake':
            for i,(x,y) in enumerate(env.body):
                pg.draw.rect(self.screen,(70,220,140) if i else (160,255,190),(x*20,y*20,19,19))
            if env.food is not None:
                x,y=env.food;pg.draw.rect(self.screen,(240,80,80),(x*20,y*20,19,19))
        elif env.name=='pong':
            pg.draw.line(self.screen,(70,70,90),(400,0),(400,500),2)
            self.rect(env.player,(90,220,140));self.rect(env.opponent,(240,180,70));self.rect(env.ball,(240,240,240))
        elif env.name=='breakout':
            colors=[(230,80,80),(240,160,60),(240,220,80),(90,200,120),(80,150,230)]
            for i in np.flatnonzero(env.hp):
                b=env.bricks[i];self.rect(b,colors[i//10])
                if env.hp[i]>1:pg.draw.rect(self.screen,(235,235,235),(round(b.x),round(b.y),72,22),2)
            self.rect(env.paddle,(235,235,235));self.rect(env.ball,(235,235,235))
        else:
            self.rect(env.ship,(90,220,120))
            for x,y in env.aliens:pg.draw.rect(self.screen,(230,90,200),(round(x),round(y),36,24),border_radius=4)
            for x,y in env.shields:pg.draw.rect(self.screen,(60,160,90),(round(x),round(y),10,10))
            for s in env.shots:self.rect(s,(240,240,240))
            for s in env.enemy_shots:self.rect(s,(250,120,80))
        info=env.info();stats=f"{env.name} | {mode} | score {info['score']} | passo {info['steps']}"
        if 'lives' in info:stats+=f" | vidas {info['lives']}"
        if 'wave' in info:stats+=f" | onda {info['wave']}"
        self.screen.blit(self.font.render(stats,True,(255,255,255)),(10,8))
        label=f'acao: {action}' if action is not None else ''
        if isinstance(agent,QLearning):
            from .agents import state_index
            label+=f' | Q max: {agent.Q[state_index(env.observe())].max():.2f}'
        elif isinstance(agent,DQN):label+=f' | Q max: {agent.q(env.observe()).max():.2f}'
        if inference_us is not None:label+=f' | inferencia: {inference_us:.1f} us'
        self.screen.blit(self.font.render(label,True,(180,190,210)),(10,32))
        help_text='A: IA  H: humano  R: reiniciar  Esc: sair | setas e espaco'
        self.screen.blit(self.font.render(help_text,True,(180,190,210)),(10,env.height-26))
        if env.ended:
            txt=self.big.render(f'{env.reason} — R para reiniciar',True,(255,240,180))
            self.screen.blit(txt,txt.get_rect(center=(env.width/2,env.height/2)))
        pg.display.flip()
    def close(self):self.pg.quit()

class Session:
    """Eventos testáveis sem simular teclas na lógica física."""
    def __init__(self,env,agent,seed=0):
        self.env,self.agent,self.seed=env,agent,seed;env.reset(seed)
        self.renderer=Renderer(env);self.pg=self.renderer.pg
        self.mode,self.running,self.pending='IA',True,0
        self.last_action=0;self.inference_us=0.
    def event(self,event):
        pg=self.pg
        if event.type==pg.QUIT:self.running=False
        elif event.type==pg.KEYDOWN:
            if event.key==pg.K_ESCAPE:self.running=False
            elif event.key==pg.K_a:self.mode='IA'
            elif event.key==pg.K_h:self.mode='humano'
            elif event.key==pg.K_r:self.seed+=1;self.env.reset(self.seed);self.pending=0
            elif self.env.name=='snake' and self.mode=='humano':
                desired={pg.K_RIGHT:0,pg.K_DOWN:1,pg.K_LEFT:2,pg.K_UP:3}.get(event.key)
                if desired is not None:
                    delta=(desired-self.env.direction)%4
                    self.pending={0:0,1:2,3:1}.get(delta,0)
    def human_action(self,keys):
        pg=self.pg
        if self.env.name=='snake':a=self.pending;self.pending=0;return a
        if self.env.name=='pong':return 1 if keys[pg.K_UP] else 2 if keys[pg.K_DOWN] else 0
        action=1 if keys[pg.K_LEFT] else 2 if keys[pg.K_RIGHT] else 0
        return action+3 if self.env.name=='invaders' and keys[pg.K_SPACE] else action
    def advance(self):
        if self.env.ended:return
        import time
        start=time.perf_counter_ns()
        action=self.agent.act(self.env.observe()) if self.mode=='IA' else self.human_action(self.pg.key.get_pressed())
        self.inference_us=(time.perf_counter_ns()-start)/1000
        self.env.step(action);self.last_action=action
    def draw(self):self.renderer.draw(self.env,self.agent if self.mode=='IA' else None,self.mode,self.last_action,self.inference_us)
    def close(self):self.renderer.close();self.env.close()

def choose_game():
    try:import pygame as pg
    except ImportError:raise RuntimeError('Instale pygame-ce para abrir o menu') from None
    pg.init();screen=pg.display.set_mode((800,500));font=pg.font.Font(None,40);clock=pg.time.Clock()
    games=['snake','pong','breakout','invaders']
    while True:
        for event in pg.event.get():
            if event.type==pg.QUIT or event.type==pg.KEYDOWN and event.key==pg.K_ESCAPE:pg.quit();return None
            if event.type==pg.KEYDOWN and event.key in [pg.K_1,pg.K_2,pg.K_3,pg.K_4]:
                game=games[[pg.K_1,pg.K_2,pg.K_3,pg.K_4].index(event.key)];pg.quit();return game
        screen.fill((15,15,25))
        for i,line in enumerate(['IA para Jogos 2D','Selecione com 1–4; Esc para sair',*[f'{j+1}. {g}' for j,g in enumerate(games)]]):
            screen.blit(font.render(line,True,(220,235,250)),(80,60+i*60))
        pg.display.flip();clock.tick(60)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--game',choices=['snake','pong','breakout','invaders'])
    p.add_argument('--model',type=Path);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--frames',type=int,help='Limite de frames para QA');p.add_argument('--fast',action='store_true',help='QA: um step por frame, sem espera')
    p.add_argument('--screenshot',type=Path,help='Salvar frame final para inspeção')
    args=p.parse_args(argv)
    if args.fast and args.frames is None:p.error('--fast requer --frames')
    game=args.game or choose_game()
    if game is None:return
    agent=load_agent(args.model,game)[0] if args.model else Heuristic(game)
    env=make_env(game);session=Session(env,agent,args.seed);clock=session.pg.time.Clock();accum=0.;frames=0
    try:
        while session.running and (args.frames is None or frames<args.frames):
            elapsed=env.dt if args.fast else min(clock.tick(60)/1000,.25)
            for event in session.pg.event.get():session.event(event)
            accum+=elapsed
            while accum>=env.dt and session.running:session.advance();accum-=env.dt
            session.draw();frames+=1
        if args.screenshot:
            args.screenshot.parent.mkdir(parents=True,exist_ok=True);session.pg.image.save(session.renderer.screen,str(args.screenshot))
    finally:session.close()

if __name__=='__main__':main()
