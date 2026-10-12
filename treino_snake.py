from pathlib import Path
from games2d.experiment import train_agent, PROFILES
from games2d.agents import save_agent
    
# 1. Configuração do jogo e candidato:
GAME = 'snake'
CANDIDATE = 'genetic'
OUT_FILE = Path('genetic-snake.json')
    
# 2. Orçamento para Algoritmo Genético:
config = dict(PROFILES['pilot'])
config['ga_population'] = 30     # Tamanho da população de cobras
config['ga_generations'] = 20    # Número de gerações para evoluir
config['ga_elite'] = 5           # Melhores indivíduos que sobrevivem sem mutação
config['ga_games'] = 5           # Partidas por indivíduo para tirar a média do fitness

# 3. Treina em CPU (modo headless acelerado):
print(f"Treinando {GAME} com {CANDIDATE}...")
agent, meta, trace = train_agent(GAME, CANDIDATE, seed=42, config=config)

# 4. Salva o checkpoint:
save_agent(agent, OUT_FILE, metadata=meta)
print(f"Modelo salvo em {OUT_FILE}")
