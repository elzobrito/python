# Benchmark games2d

Perfil: **pilot**. 5 treinamentos por candidato; 100 partidas por modelo.

Pontuação original; shaping excluído da avaliação. IC95 de média por bootstrap hierárquico. Diferenças pareadas usam IC97,5 para duas comparações do candidato selecionado exclusivamente pela validação.

Orçamentos e horizonte constam em config.json. Tempo de treino não significa convergência. Latência inclui overhead da chamada/timer; p95 reportado é o máximo entre runs.

| Jogo | Agente | Média | IC95 | Mediana | Desvio | p95 µs | CPU treino s | Truncamentos |
|---|---|---:|---|---:|---:|---:|---:|---:|
| snake | random | 0.68 | [0.32, 1.14] | 0.00 | 2.82 | 2.1 | 0.00 | 69.0% |
| snake | heuristic | 366.90 | [345.20, 388.10] | 370.00 | 109.50 | 36.9 | 0.00 | 32.0% |
| snake | q_plain | 291.20 | [247.01, 345.08] | 310.00 | 124.45 | 43.5 | 59.96 | 34.8% |
| snake | q_shaping | 0.06 | [0.00, 0.20] | 0.00 | 0.77 | 38.3 | 22.23 | 61.0% |
| snake | genetic | 3.40 | [0.98, 6.20] | 0.00 | 6.43 | 1.9 | 0.05 | 13.6% |
| pong | random | -3.69 | [-3.82, -3.56] | -4.00 | 0.98 | 6.9 | 0.00 | 74.8% |
| pong | heuristic | 0.00 | [0.00, 0.00] | 0.00 | 0.00 | 0.8 | 0.00 | 100.0% |
| pong | cart | -3.69 | [-3.86, -3.52] | -4.00 | 0.91 | 5.2 | 1.30 | 79.6% |
| pong | dqn | -4.07 | [-4.26, -3.85] | -4.00 | 0.83 | 29.5 | 0.35 | 63.4% |
| breakout | random | 80.36 | [76.00, 85.20] | 60.00 | 28.86 | 5.7 | 0.00 | 0.0% |
| breakout | heuristic | 240.00 | [240.00, 240.00] | 240.00 | 0.00 | 0.8 | 0.00 | 100.0% |
| breakout | dqn | 76.00 | [68.00, 80.00] | 80.00 | 8.01 | 30.3 | 0.57 | 0.0% |
| invaders | random | 218.74 | [213.54, 223.48] | 220.00 | 36.92 | 4.4 | 0.00 | 95.8% |
| invaders | heuristic | 82.00 | [80.50, 83.60] | 80.00 | 7.88 | 5.0 | 0.00 | 100.0% |
| invaders | genetic | 102.66 | [101.72, 103.76] | 100.00 | 5.21 | 94.8 | 13.04 | 100.0% |

## Decisões

- snake: **SEM_VENCEDOR_COMPROVADO**; candidato escolhido na validação: `q_plain`.

- pong: **SEM_VENCEDOR_COMPROVADO**; candidato escolhido na validação: `cart`.

- breakout: **SEM_VENCEDOR_COMPROVADO**; candidato escolhido na validação: `dqn`.

- invaders: **SEM_VENCEDOR_COMPROVADO**; candidato escolhido na validação: `genetic`.

## Limitações

Observações Snake/Invaders são parciais. Horizons e física são variantes documentadas. Ruído de coleta não garante recuperação da imitação. Smoke não prova eficácia; pilot usa treino reduzido. Avaliação automatizada não representa aceite humano nem revisão independente.

## Artefatos

config.json, provenance.json, costs.json, models.json, training.csv, validation.csv, evaluation.csv, summary.json, artifact_hashes.json e modelos representativos selecionados pela validação.
