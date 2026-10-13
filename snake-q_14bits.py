from pathlib import Path
from games2d.experiment import train_agent, PROFILES
from games2d.agents import save_agent

# 1. Configura o perfil e ativa os 14 bits:
config = dict(PROFILES['pilot'])
config['snake_bits'] = 14        # Ativa o sensor de 14 bits!
config['q_episodes'] = 10000      # 10.000 episódios (~3 a 5 min em CPU)

out_file = Path('snake-q_14bits.json')

# 2. Treina em CPU pura:
print("Treinando Snake com 14 bits (16.384 estados)...")
agent, meta, trace = train_agent('snake', 'q_plain', seed=42, config=config)
meta['bits'] = 14

# 3. Salva o checkpoint:
save_agent(agent, out_file, metadata=meta)
print(f"Modelo salvo em {out_file}! Tabela Q: {agent.Q.shape}")
