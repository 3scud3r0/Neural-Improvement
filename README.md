# NEXUS AutoSynth — síntese de algoritmos por MCTS sem LLM

**Novo laboratório executável: AST Python, busca MCTS sobre programas, benchmarks isolados, verificação finita, melhoramento iterativo de fonte, aprovação, hot-swap e rollback.** Mantém todos os módulos da versão anterior NEXUS Evolution Lab.

## Início rápido

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m nexus_m.autosynth_cli search --simulations 96 --output runs/autosynth
```

O comando **não** promove código sem `--approve`. Para analisar a proposta gerada: `runs/autosynth/managed/candidate_ast_router.py`, `proof_evidence.json`, `ast_search_trace.json`. Após inspecionar, execute `search --approve` para promover apenas uma implementação verificada no diretório gerenciado. Para reverter: `rollback --approve`.

A gramática atual gera 24 programas AST de otimização top-k, **não** qualquer programa Python; não existem provas universais de alinhamento, execução nativa autônoma, provisionamento de nuvem ou autoaperfeiçoamento ilimitado.

![Resultados medidos do experimento AutoSynth](outputs/autosynth/autosynth_resultados.png)

Leia [Arquitetura e implementação](docs/AUTOSYNTH_ARQUITETURA.md), [Resultados medidos](docs/RESULTADOS_AUTOSYNTH.md) e [Limitações de prova formal](formal/OBRIGACOES_DE_PROVA.md).

---

# NEXUS Evolution Lab — arquitetura modular para pesquisa em inteligência adaptativa

**Estado: protótipo científico executável. NÃO é AGI, Singularidade Tecnológica,
autoaperfeiçoamento ilimitado nem uma prova de alinhamento universal.**

Projeto desenvolvido a partir do **NEXUS-M original**. Os módulos anteriores
(`grammar.py`, `mcts.py`, `model.py`, `memory.py`, `world_model.py`, treinamento,
telemetria, precisão decimal) continuam presentes e funcionando. Os novos
arquivos são **módulos alternativos e complementares**, sem quebrar o legado.

## Primeiro uso

Requer Python >=3.10, PyTorch, NumPy, NetworkX e SymPy. Testado neste ambiente
com Python 3.13, PyTorch 2.10, NetworkX 3.6 e SymPy 1.14.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m nexus_m.evolution_cli demo --output runs/evolution_demo --simulations 1500
```

Os resultados ficam em `runs/evolution_demo/demo_result.json`, o conhecimento
em `knowledge.sqlite`, as atualizações em `events.jsonl` e o registro de
autorizações em `audit_chain.jsonl`.

Para sintetizar um programa a partir dos próprios pares entrada/saída:

```bash
python -m nexus_m.evolution_cli search \
    --examples examples/evolution_problem.json \
    --simulations 1500 --depth 6 \
    --output runs/my_problem
```

A busca nunca recebe o programa gerador nem os valores de saída dos exemplos
reservados para avaliação. Entradas adicionais sem rótulo são opcionais.

## Sete componentes principais

| Módulo | Implementação executável | Limite explícito |
|---|---|---|
| `universal_compute.py` | Interpretador com inteiros exatos, `while`, condicionais, limite de instruções; equivalência finita, prova polinomial e adaptadores opcionais Lean/Coq/compiladores | Não executa arbitrariamente Python/C++/Rust; provas finitas não são gerais; compiladores exigem opt-in |
| `hierarchical_abstraction_search.py` | PUCT em macroações, busca com expansão progressiva, grafo de subobjetivos, controle de profundidade | Domínio de transformações inteiras; não resolve problemas científicos abertos |
| `dynamic_topology.py` | Crescer/reduzir módulos, aumentar neurônios, preservar pesos sobrepostos e limitar recursos | Sem aquisição autônoma de GPUs/serviços; exige novo otimizador após mutações |
| `causal_inference_engine.py` | SCM binário, `do(X=x)`, ajustes backdoor/frontdoor, regressão física e testes de intervenção | Não implementa do-calculus completo nem física universal |
| `conceptual_knowledge_graph.py` | SQLite persistente, grafos tipados, relações com evidência, URIs/hash multimodais e isomorfismos estruturais | Analogia não garante transferência real; não baixa toda a internet |
| `recursive_self_improvement.py` | Perfil de algoritmo, alternativas revisadas, comparação, equivalência, arquivo gerenciado, rollback e troca de função em execução | Até 5 ciclos por processo, apenas plugin allowlisted, aprovação humana por padrão |
| `alignment_and_value_core.py` | Quotas, permissões, revisão humana, trilha de auditoria com hashes e negação de capacidades não autorizadas | **Não** resolve alinhamento de valores nem torna o sistema incorruptível |

## Componentes adicionais integrados

- `latent_world_model.py`: modelo preditivo em espaço latente, inspirado em
  JEPA, com codificador-alvo atualizado por EMA, treinamento e penalidade de
  colapso; **apenas** domínio de operações sintéticas.
- `meta_learning.py`: destilação de programas consistentes com exemplos
  encontrados na busca para os pesos dos 100 agentes originais; limite de
  passos e opção de atualização temporária.
- `distributed_coordinator.py`: múltiplos processos locais com árvores MCTS
  independentes, sementes distintas e agregação de resultados.
- `evolution_system.py`: orquestrador conecta busca, especialistas neurais,
  memória, modelo causal, modelo latente, plasticidade e controles.
- NEXUS-M original: treinamento supervisionado de 100 agentes de 100/1.000/
  10.000 neurônios, telemetria de pesos, FP32/FP64, microbackend Decimal com
  100 dígitos, MCTS de RPN e checkpoints.

## Benchmark experimental executado

```bash
python -m scripts.benchmark_evolution \
    --seeds 2026 2027 2028 --simulations 350 \
    --output outputs/evolution_benchmark
python -m scripts.plot_evolution
```

Resultados reais nesta execução (24 problemas sintéticos, 3 sementes):

| Condição | Soluções exatas em teste | Simulações MCTS médias |
|---|---:|---:|
| Política uniforme + MCTS | 20/24 | 146,2 |
| Política neural **não treinada** + MCTS | 15/24 | 198,1 |
| Preditor latente + MCTS | 23/24 | 80,0 |
| Sistema coletivo, memória e destilação online | 20/24 | 141,3 |

Os resultados estão em `outputs/evolution_benchmark/{summary.json,results.csv}`.
O preditor latente foi o melhor nesta pequena amostra, mas as condições
**não têm o mesmo custo total de treinamento** e os problemas compartilham a
mesma gramática. Nenhuma conclusão de AGI ou de generalização ampla é válida.

## Melhoramento de código com revisão

```bash
# Apenas propor, avaliar e documentar; nenhuma alteração é aplicada:
python -m nexus_m.evolution_cli improve --output runs/improve

# Promover uma das duas alternativas predefinidas SOMENTE se os testes e
# requisitos mínimos de ganho passarem, com consentimento explícito:
python -m nexus_m.evolution_cli improve --output runs/improve --approve
```

Uma versão futura pode usar novas propostas geradas por modelos, mas exigiria
isolamento de processos real, avaliação muito mais robusta e revisão humana.
A rotina atual **não edita arquivos arbitrários**, não altera o guardião de
segurança e não cria um ciclo infinito de autoaperfeiçoamento.

## MCTS distribuído em CPU local

```python
from nexus_m.distributed_coordinator import DistributedSearchCoordinator
x = [-4, -1, 0, 2, 5]
examples = [(v, 2*v+3) for v in x]
result = DistributedSearchCoordinator(workers=2).search(
    examples, simulations_per_worker=350, depth=6)
print(result['found'], result['winner'])
```

## Interpretação de universalidade e de provas

O subconjunto interpretado de Python admite contador, condição e repetição,
o suficiente para expressar construções de computação universal em princípio.
**Todas as execuções são finitas**, porque o interpretador limita instruções,
tamanho numérico e tempo. Provas gerais sobre programas arbitrários são
limitadas pela indecidibilidade do problema da parada e pela necessidade de
especificações precisas. O sistema fornece provas simbólicas somente para
identidades polinomiais e verificações exaustivas em domínios finitos.

Os adaptadores Lean/Coq podem chamar provadores instalados com opt-in; a
presença de tal adaptador não significa provar a correção de qualquer
programa, muito menos de mudanças futuras. Compiladores C++/Rust/Assembly
podem produzir diagnósticos, mas não são um ambiente seguro de execução e
**não executamos seus binários**.

## Recursos e precisão

Os limites padrão são 1.000 redes, 10.000 neurônios por rede, 25 milhões
de parâmetros e 100.000 simulações MCTS por operação. Nenhum recurso de nuvem
é contratado e não há acesso de rede habilitado por padrão. Os limites não
substituem containerização/quotas do sistema operacional.

Para telemetria de estados, mensagens, gradientes e pesos do modelo original:

```bash
python -m nexus_m train --agents 100 --width 100 \
    --steps 20 --precision float64 --telemetry full \
    --output runs/telemetry
python -m nexus_m decimal-demo --digits 100 --output runs/decimal.json
```

A precisão decimal alta é usada no **microbackend**; ela não transforma os
pesos de 1 milhão de neurônios em números de 100 dígitos nem garante melhor
inteligência.

## Documentação

- `docs/MODULOS_EVOLUCAO.md` — APIs e limites de cada um dos novos componentes.
- `docs/SEGURANCA_E_LIMITES.md` — políticas, riscos e modelo de ameaças.
- `docs/VERIFICACAO_FORMAL.md` — interpretação e provas reais versus alegações.
- `docs/EXPERIMENTO_EVOLUCAO.md` — metodologia e comparações executadas.
- `docs/ROADMAP_CIENTIFICO.md` — pesquisas que ainda seriam necessárias.
- `docs/README_NEXUS_M_ORIGINAL.md` — README integral do sistema anterior.

## Limites essenciais

Não existe atualmente uma implementação aceita de AGI universal garantida,
autoaperfeiçoamento recursivo infinito ou alinhamento humano incorruptível.
Este projeto **não** afirma obtê-los. O benefício demonstrado está no código
modular executável, experimentos reproduzíveis e na avaliação explícita do
que funciona e do que falha.
