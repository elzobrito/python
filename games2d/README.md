# IA para Jogos 2D

Pacote adicional aos 30 algoritmos didáticos. Inclui Snake, Pong, Breakout e
Space Invaders; agentes aleatório/heurístico, Q-learning com/sem shaping,
CART de arvores.py, DQN contínuo e evolução de controladores lineares/utilidade.
Nenhuma alteração em g1–g6 nem nos jogos-fonte da disciplina.

## Demonstração no desktop

Veja o [screenshot do menu e o vídeo da demonstração](../README.md#demonstração-dos-jogos-2d)
no README principal. As mídias estão incluídas no repositório.

## Executar

Execute os comandos na pasta `/home/elzobrito/desenvolvimento/algoritmos/python`.
O treino usa NumPy e biblioteca padrão; não importa Pygame nem abre display.

```bash
python -m games2d.experiment --profile smoke --out /tmp/games2d-smoke-novo
python -m games2d.experiment --profile pilot --out /tmp/games2d-pilot-novo
python -m games2d.experiment --profile research --games snake pong --out /tmp/games2d-research-novo
python -m games2d.benchmark --run games2d/results/pilot
```

Saídas de experimentos devem ser novas/vazias; não sobrescreva evidências.
O benchmark sem --out apenas verifica hashes e mostra o relatório congelado.
Para reproduzir todas as partidas dos checkpoints, sem retreinar:

```bash
python -m games2d.benchmark --run games2d/results/pilot --out /tmp/games2d-reavaliacao-nova
```

Instalação isolada, se necessária:

```bash
python -m venv .venv-games
.venv-games/bin/python -m pip install -r games2d/requirements-visual.txt
.venv-games/bin/python -m games2d.play
```

Neste notebook existe `uv`; também é possível iniciar sem criar venv no repo:

```bash
uv run --with numpy --with pygame-ce python -m games2d.play
uv run --with numpy --with pygame-ce python -m games2d.play --game snake --model games2d/results/pilot/snake-q_plain.json
```

O menu usa 1–4. A ativa IA, H ativa humano, R reinicia, Esc sai. Setas controlam
movimento; espaço dispara em Invaders. Snake bloqueia inversão de 180 graus.
Sem --model, a demonstração usa a heurística. Um arquivo de modelo de outro
jogo é rejeitado. Não há treinamento durante a demonstração.

## Perfis e interpretação

| Perfil | Runs | Testes/modelo | Q episódios | DQN steps | CART amostras | GA população × gerações × jogos | Horizonte máximo |
|---|---:|---:|---:|---:|---:|---|---:|
| smoke | 1 | 10 | 15 | 400 | 300 | 4×2×2 | 180 |
| pilot | 5 | 100 | 2000 | 3000 | 10000 | 6×3×3 | 1200 |
| research | 5 | 100 | 2000 | 20000 | 10000 | 30×20×5 | 3600 |

Snake research é limitado a 2000 steps; Invaders sempre a 1200. Snake também
trunca em 100 passos sem comer, sem penalidade de morte. Horizons curtos podem
produzir empates em Pong; devem ser considerados na leitura dos resultados.
Smoke prova execução, não eficácia. Pilot testa o protocolo com orçamento de
aprendizado menor; research não garante convergência nem campeão.

O comando --config arquivo.json aceita overrides explícitos dos campos do perfil.
Config integral é persistido antes do treino. Não use seeds de teste para ajustar
hiperparâmetros. Treino usa seeds derivadas fora do holdout; validação 500–519;
teste 1000–1099. Todos os candidatos enfrentam os mesmos limites e seeds de
ambiente. A política aleatória tem stream própria, sem perturbar RNG da física.

Modelos por run ficam agregados em models.json. Arquivos `jogo-candidato.json`
são modelos representativos escolhidos pela média de validação, nunca pelo teste.
São checkpoints para inferência, sem buffer de replay/estado completo de retomada.

O relatório traz scores originais, IC95 hierárquico, comparações pareadas,
latência mediana/p95, tempos de CPU/wall e truncamentos. O candidato escolhido
na validação só recebe suporte no teste se superar ambos os baselines com
limite inferior positivo do IC97,5 (duas comparações) e p95≤1 ms, com ≥5 runs
/100 partidas por modelo. Isso é evidência dentro deste orçamento e variante,
não uma declaração universal de algoritmo ideal. Ausência de suporte produz
SEM_VENCEDOR_COMPROVADO. Critérios não usam razões com scores negativos nem CV.

## Arquitetura e variantes

- env_base.py: ABC e retângulos independentes de Pygame.
- environments.py: reset/step/terminated/truncated e física dos quatro jogos.
- agents.py: políticas, Bellman, DQN/replay/target/Huber, CART e genética.
- experiment.py: coleta, treino, validação, teste e persistência.
- benchmark.py: reavaliação, intervalos, latência, relatório e curvas opcionais.
- play.py: menu, renderer e controles A/H/R, usando o mesmo núcleo físico.
- PROTOCOL.md: especificação congelada antes da implementação.

As variantes não prometem equivalência bit a bit com os scripts originais:
movimento de raquetes/nave usa coordenadas subpixel; dt fixo substitui o tempo
variável de renderização; Snake tem corpo/cauda/comida corretos e ações relativas;
Pong tem teto de velocidade. Essas regras são comuns ao treino e renderer.
Observações Snake e Invaders são parciais. Potencial Snake usa distância interna
e zero terminal; isso não garante convergência da representação de 11 bits.
Pong usa centros das raquetes e canto superior esquerdo da bola; heurística
compensa meia altura da bola. Breakout mantém slots fixos para resistência 0/1/2.
Invaders tem seis ações e escudos que bloqueiam ambos os lados; a heurística
pode abrir passagem disparando no escudo. Genética usa features específicas
por ação. Os nomes dos operadores referem-se aos mecanismos do repositório;
as demonstrações originais não são chamadas como agentes completos.

Matplotlib é opcional; quando presente gera curvas SVG exportáveis. Sem ele,
todos os dados das curvas continuam nos CSVs. Pygame-ce fornece o módulo
`pygame`, com wheel para Python 3.14. Dependências visuais não são necessárias
para treino. Testes visuais usam SDL dummy, não provam aceite humano em desktop.

## Verificar

```bash
python -m pytest -q tests/test_games2d.py
SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy .venv-games/bin/python -m pytest -q tests/test_games2d.py
python run_all.py
```

Sem pygame, o teste visual é marcado skipped; use o ambiente visual para QA
completa. Governança: tarefas G2D-20261012-001..006, reparo 007 e hotfix associado,
no ESAA de /home/elzobrito. Revisão pelo mesmo runner Codex não é independente.
Evidências da execução encontram-se em QA.md e results/pilot/REPORT.md.
