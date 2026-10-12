# Protocolo games2d v1

Escopo: Snake → Pong → Breakout → Space Invaders, em um pacote adicional.
Os exemplos g1–g6 e os jogos da disciplina são referências; não são modificados.
Algoritmos e física dependem somente de NumPy e biblioteca padrão. Pygame é
opcional e importado apenas pelo renderer. Matplotlib é opcional para curvas.

## Contrato e regras

`reset(seed) -> (observation, info)`; `step(action) -> (observation, reward,
terminated, truncated, info)`; `render()`; `close()`. Um step Snake move uma
célula (8 movimentos/s na demonstração); os demais avançam dt=1/60 s.
O renderer usa exatamente o mesmo estado físico do treino.
Sementes do ambiente, ações aleatórias e treinamento usam geradores separados.
Observações retornam cópias; ações inválidas e step após encerramento são erros.

Snake: grade 40×30; 3 ações relativas; comida só em células livres; colisão
com corpo; entrar na cauda que sai no mesmo passo é permitido. Tabuleiro cheio
é vitória. Esses são ajustes explícitos ao snake.py didático.

Sensores do Snake (11 bits padrão e extensão 14 bits):
A observação é parcial (não um MDP completo). O índice do estado soma bit[i]*2**i:
- Bits 0..2: Perigo imediato a 1 passo (frente, esquerda, direita).
- Bits 3..6: Direção absoluta do movimento (cima, direita, baixo, esquerda).
- Bits 7..10: Posição relativa da comida (acima, à direita, abaixo, à esquerda).
- Bits 11..13 (modo bits=14): Perigo antecipado a 2 passos (frente, esquerda, direita).
A extensão para 14 bits (bits=14) foi motivada pela miopia espacial do sensor de 11 bits:
ao atingir tamanho >15–20, a cobra frequentemente entrava em becos sem saída de 1 célula
ou realizava espirais fatais porque uma ação segura a 1 passo pode ser armadilha a 2 passos.
O lookahead de profundidade 2 previne o auto-encarceramento em bolsões imediatos.
O espaço discreto salta de 2^11 = 2.048 para 2^14 = 16.384 estados. A Q-table resultante
(16.384×3 = 49.152 valores float64, ~330 KB em JSON) permanece leve e treinável puramente
em CPU em poucos minutos (ex: 10.000 episódios), sem necessidade de aceleração em GPU.
O ambiente mantém bits=11 como padrão para compatibilidade retroativa com os baselines
congelados, suportando bits=14 via parâmetro explícito ou configuração.

Recompensa base: comida +10; morte -10; vitória inclui comida final.
Shaping opcional: gamma=0.9, Phi=-distância Manhattan em células, Phi terminal=0.
Phi é calculado no estado interno completo, antes/depois inclusive nova comida.
Truncamento externo após 100 passos sem comer ou 2000 passos totais: sem
penalidade artificial, bootstrap preservado; não se zera Phi em truncamento.

Pong: 800×500; parado/subir/descer; 5 pontos encerram partida. Observação de
6 valores: centros y jogador/oponente, bx, by em [0,1], vx/vy em [-1,1].
Teto explícito de |vx|=800 para todos os controles. Oponente segue a bola,
velocidade 75% da velocidade do jogador. Recompensa somente +1/-1 por ponto.
Partida truncada após 3600 steps. Score principal: saldo de pontos.

Breakout: 800×600; parado/esquerda/direita; 3 vidas, 5×10 tijolos. Resistência
0/1/2 preservada em slots estáveis, observação de 56 valores, posições e vidas
normalizadas; velocidades divididas por 1.8*340. Recompensa por dano +1,
perda de vida -2, vitória +10. Score original: 10 por impacto +10 por destruição.
Fim com zero vidas ou tijolos, truncamento em 3600 steps. Menor sobreposição
resolve colisão; física angular e teto de velocidade mantidos.

Invaders: 800×600; 6 ações (parado/esquerda/direita e cada uma com disparo).
3 vidas, cooldown .35 s, 50 invasores, escudos de blocos que interceptam tiros
amigos e inimigos, novas ondas. 9 atributos táticos normalizados: nave,
cooldown, xmin,xmax,ybase,dx da formação, projétil ameaçador dx/dy, integridade
local do escudo. Observação parcial; utilidade consulta somente esses atributos.
Fim por vidas/base, horizonte externo 1200 steps. Score original: 10 por abate.
Recompensa por abate +1, dano -5, base -10. Uma ameaça é escolhida por tempo
até alcançar a nave, priorizando tiros em sua faixa horizontal.

## Candidatos e budgets (não são previsões de convergência)

Todos: aleatório e heurística, com sementes de política distintas das ambientais.
Snake: Q-learning com/sem shaping (alpha .1, gamma .9, eps 1→.01), genético
linear 11×3. Pong: CART e DQN 6→16→3. Breakout: DQN 56→32→3.
Invaders: utilidade evolutiva com features dependentes da ação.
Q: 2000 episódios por treinamento. DQN: 20000 transições, warmup 256,
minibatch 32, replay circular 10000, treino a cada 4 steps, target a cada 200,
lr .001, Huber loss e clipping de gradiente. GA: 30 indivíduos, 20 gerações,
5 partidas por indivíduo, elites 10, crossover uniforme e mutação gaussiana.
Fitness Snake: maçãs primeiro (100*maçãs), penalidade pequena por steps;
Invaders: pontuação original, média sobre cinco partidas. Não há bônus de vida.
CART: 10000 transições, partidas de coleta sem dados do teste; perturbações
fazem o ambiente visitar estados de recuperação, rótulo vem da heurística no
estado perturbado. Validação por partidas independentes, não por quadros.

## Protocolo experimental

5 treinamentos independentes seeds 0..4; validação 500..519; teste 1000..1099.
Uma stream por seed de treino deriva seeds de episódios sem acessar validação
ou teste. Hiperparâmetros são congelados no config antes de avaliar o teste.
Nenhuma seleção de checkpoint pelo teste. Treino interrompido por budget,
não rotulado como convergência. Smoke é só prova de execução e não eficácia.

Métricas: score original, retorno base, retorno shaped separado, steps, motivo
de encerramento, vitória (não aplicável a ondas infinitas), maçãs/vidas/ondas.
Relatar médias, medianas, desvio, intervalo bootstrap hierárquico de 95%:
reamostrar treinamentos e seeds de avaliação comuns, não tratar todas as
partidas como evidências independentes da estabilidade de treino.
Comparações pareadas com os baselines no mesmo conjunto de seeds; streams
ambientais são reprodutíveis mas políticas diferentes não criam trajetórias iguais.
Latência: aquecimento, perf_counter_ns, mediana/p95, hardware/versões registrados.
Tempo wall e CPU do treino, número de transições e updates registrados.
Recomendação requer diferença inferior do IC >0 contra heurística e aleatório,
p95≤1 ms, validação de contrato; caso contrário SEM_VENCEDOR_COMPROVADO.
Não usar múltiplos de score negativo/zero nem coeficiente de variação.

Perfis: smoke (1 treinamento/10 partidas, budgets mínimos), pilot (5/100,
budgets reduzidos declarados), research (5/100, budgets acima). Pilot permite
benchmark completo com orçamento curto, sem confundir com research.
Resultados trazem config, hashes das fontes/modelos, CSVs brutos e relatório.
Avaliação humana e uso em sala são etapas distintas da QA automatizada.

## Referências

- Ng, Harada, Russell (1999): https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf
- Mnih et al. (2013): https://arxiv.org/abs/1312.5602
- Ross et al. (2011): https://proceedings.mlr.press/v15/ross11a.html
- Agarwal et al. (2021): https://arxiv.org/abs/2108.13264
- Terminated/truncated: https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/
