from pathlib import Path
from games2d.experiment import train_agent, PROFILES
from games2d.agents import save_agent
    
# 1. Configuração do jogo e candidato:
GAME = 'snake'
CANDIDATE = 'q_plain'            # Q-Learning sem shaping
OUT_FILE = Path('q_plain-snake.json')
    
# 2. Orçamento para Q-Learning:
config = dict(PROFILES['pilot'])
config['q_episodes'] = 100000     # Aqui sim: 10.000 episódios de treino!
    
# 3. Treina em CPU (modo headless acelerado):
print(f"Treinando {GAME} com {CANDIDATE}...")
agent, meta, trace = train_agent(GAME, CANDIDATE, seed=42, config=config)
    
# 4. Salva o checkpoint:
save_agent(agent, OUT_FILE, metadata=meta)
print(f"Modelo salvo em {OUT_FILE}")