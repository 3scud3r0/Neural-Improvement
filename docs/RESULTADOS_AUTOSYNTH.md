# Resultados medidos: síntese AST + MCTS

**Ambiente:** Python 3.13, processo CPU deste ambiente, semente inicial 2026. Microbenchmarks são sensíveis a carga da CPU, tamanhos e distribuição de pontuações. Não são estimativas universais.

| Experimento | Resultado medido |
|---|---|
| ASTs possíveis nesta gramática | 24 |
| MCTS inicial | 70 simulações, 16 ASTs avaliadas |
| Melhor candidato na busca inicial | `heap`, chave estável, validação estrita |
| Ganho da primeira busca (casos da busca) | 59,17% de redução do tempo |
| Ganho independente da primeira busca | 67,32% de redução do tempo |
| Casos finitos conferidos pela seleção final | 10.467 |
| Promoção automática, sem consentimento | Não; fonte preparada para revisão |
| Ensaio separado com aprovação | Promovido `heap`, chave `score`, validação estrita |
| Ganho independente neste ensaio | 76,43% de redução do tempo em relação à ordenação |
| Nova rodada contra a versão já promovida | Ganho de 1,87%; promoção recusada por ser inferior a 5% |
| Rollback aprovado | Funcionou; fonte inicial restaurada |
| Verificação universal em Lean ou Coq | **Não executada; sem prova universal** |
| Testes automatizados integrados | 45 aprovados (37 existentes + 8 novos) |

**Interpretação:** o sistema *seleciona* algoritmos compostos de operadores conhecidos pela gramática, compila um AST de Python, testa e escolhe candidatos mais rápidos para um domínio específico. A melhoria é no kernel de seleção top-k, não uma medida de inteligência geral. Uma política uniforme de seleção poderia possivelmente encontrar a mesma otimização neste espaço muito pequeno; o MCTS será mais importante quando o espaço crescer e oferecer sinais de valor aproveitáveis.

**Seleção exata:** `heapq.nsmallest(k, range(n), key=lambda i:...)` encontra os melhores `k` índices, e empates precisam ser resolvidos pela ordem dos índices. Para a seleção com `key=lambda i: -scores[i]`, a estabilidade de `heapq.nsmallest` no iterável `range(n)` preserva a ordem dos índices empatados. O sistema não assume isto sem testes finitos e diferenciais.

**Dados auditáveis:**

- `outputs/autosynth_trial/managed/ast_search_trace.json`: histórico de 70 simulações e avaliações de candidatos.
- `outputs/autosynth_trial/managed/proof_evidence.json`: evidência finita e estrutura de terminação.
- `outputs/autosynth_trial/managed/heldout_measurement.json`: avaliação reservada.
- `outputs/autosynth_approved_example_report.json`: execução com aprovação.
- `outputs/autosynth_second_iteration_report.json`: segunda rodada contra baseline atualizado.
- `outputs/autosynth_rollback_report.json`: reversão.
- `outputs/autosynth/autosynth_resultados.png`: visualização gerada do histórico.

**Limites:** não foram gerados milhões ou bilhões de variantes; não foi feito controle de precisão contra ruído de sistema operacional em múltiplas máquinas; testes finitos não provam ausência de bugs para todos os inteiros ou todos os programas Python; a promoção não altera módulos principais nem credenciais.
