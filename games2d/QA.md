# Validação games2d — 2026-10-12

Implementação concluída dos quatro ambientes, baselines, Q-learning com/sem
shaping, CART, DQN, evolução de controladores, persistência, experimentos e
renderer comum. Efeitos finais aplicados pelo ESAA de /home/elzobrito.

## Evidências executadas

- 26 testes passaram no runtime visual isolado; Python padrão: 25 passaram e
  1 teste visual skipped por falta de pygame. Testes em `../tests/test_games2d.py`.
- `python -B run_all.py`: exit 0, 139 linhas de saída; 13 arquivos originais
  conferidos por SHA-256 e preservados. Nenhuma alteração em g1–g6/arvores/dados.
- Perfil pilot: 75 registros de modelo, 1500 partidas de validação e 7500 de
  teste; cinco runs por candidato, 100 testes por modelo.
- 45 partidas amostradas (3 seeds × 15 candidatos) reproduzidas exatamente
  após recarga JSON, incluindo score, retorno, passos e encerramento.
- Fontes físicas/agentes e modelos preservados durante o benchmark; correção
  posterior apenas na apresentação de séries curtas dos gráficos, registrada
  aditivamente em provenance.json. CSVs, modelos e summary.json mantiveram hashes.
- Renderer SDL dummy dos quatro jogos: pixels reais inspecionados, A/H/R/Esc
  automatizados; menu 1–4 verificado por eventos. Modelo treinado Snake carregado
  e renderizado por 20 passos com Q max e latência no HUD.
- Gráficos SVG rasterizados e inspecionados; correção de séries curtas registrada
  por ISS-G2D-CURVES-001/G2D-20261012-008. Heurística de Invaders corrigida por
  ISS-G2D-INVADERS-HEURISTIC-001/G2D-20261012-007 para abrir passagem nos escudos.

## Resultado experimental

Todos os jogos: **SEM_VENCEDOR_COMPROVADO** entre os candidatos aprendidos.
A política aleatória foi incluída, sem ser promovida como solução de produção.

| Jogo | Heurística: média | Melhor candidato aprendido na validação: média no teste |
|---|---:|---:|
| Snake | 366,90 | Q sem shaping: 291,20 |
| Pong | 0,00 | CART: -3,69 |
| Breakout | 240,00 | DQN: 76,00 |
| Invaders | 82,00 | Genético: 102,66 (aleatório: 218,74) |

Todas as latências p95 ficaram abaixo de 1 ms; maior valor observado: 94,8 µs.
Melhor custo de inferência não significa melhor política. Snake com shaping
obteve média 0,06: este experimento não sustenta seu uso nesta representação.
O resultado não demonstra uma falha do teorema de shaping; observação comprimida
continua parcial. Não foi realizado diagnóstico causal adicional da aprendizagem.

Ver [relatório completo](results/pilot/REPORT.md), [curvas Snake](results/pilot/snake-learning.svg)
e [curvas gerais](results/pilot/curves.svg). CSVs e modelos ficam no mesmo diretório.
O painel CART nas curvas gerais registra um fit único, com score sentinela zero;
não é uma curva de eficácia. Sua eficácia está em evaluation.csv/REPORT.md.

## Limites de aceite

Pilot tem budgets menores que research e horizonte curto; não há alegação de
convergência. Perfil research está implementado, mas não foi executado nesta QA.
Revisão ESAA pelo mesmo runner Codex: não independente nem cega. A renderização
foi validada em SDL dummy; aceite humano no desktop e validação de uso em sala
não foram realizados. Resultados negativos não foram convertidos em campeão.

## Governança

Tarefas G2D-20261012-001..006; reparos 007/008 e hotfixes associados. Claim,
complete e review separados; arquivos finais via file_updates; verify após
escritas. Hotfixes automáticos de escopo genérico foram fechados com evidências
das tarefas de reparo escopadas, sem criar arquivos fora do destino autorizado.
