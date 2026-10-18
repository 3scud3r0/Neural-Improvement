# NEXUS AutoSynth — da árvore MCTS ao código Python (sem LLM)

## Objetivo

Investigar se um sistema pequeno pode **descobrir melhorias reais de algoritmos por busca de programas**, sem modelo de linguagem, mantendo testes independentes e a capacidade de restaurar a versão anterior. Este projeto **não** demonstra inteligência geral, autoaperfeiçoamento ilimitado, criação autônoma de novas arquiteturas Transformer ou prova completa de correção de qualquer código Python.

## Pipeline executável

```text
cProfile -> gargalo de roteamento top-k
    |
    v
MCTS de esboços tipados: estratégia × chave × validação (24 ASTs)
    |
    v
Gerador de AST Python (ast.FunctionDef, Call, For, Lambda, Return)
    |
    v
Compilação sintática + AST canônica conferida
    |
    v
Processo filho: casos adversariais, equivalência diferencial e tempo
    |
    v
Busca MCTS atualiza recompensa de rapidez, respeitando correção
    |
    v
Validação independente: entradas inéditas + 10.467 exemplos finitos
    |
    v
Registro do programa candidato e evidências em managed/
    |
    +-- sem --approve --> apenas sugestão, fonte ativa intacta
    |
    +-- com --approve e ganho >= 5% --> backup + os.replace + hot-swap
    |                                          |
    |                                          v
    |                                     rollback aprovado
    v
Próxima rodada: comparação contra a nova versão ativa, não contra a original
```

## Arquivos novos

| Módulo | Papel |
|---|---|
| `ast_program_synthesis.py` | Compilador de 24 ASTs válidas de roteamento top-k com três dimensões composicionais: `strategy`, `key`, `validation` |
| `ast_mcts.py` | Árvore UCB de decisões parciais sobre a gramática AST e cache das avaliações |
| `ast_candidate_runtime.py` | Casos adversariais, benchmarks, processos filhos com tempo e memória limitados |
| `ast_runtime_harness.py` | Executa somente fontes canônicas geradas pela DSL e retorna métricas em JSON |
| `ast_formal_gate.py` | Verificação exaustiva em domínio finito e evidência estrutural de terminação; **não** é uma prova Lean universal |
| `ast_performance_profiler.py` | `cProfile` do componente gerenciado para localizar gargalos reais |
| `search_driven_improver.py` | MCTS + gate + staging + autorização + hot-swap + rollback |
| `capacity_planner.py` | Planejamento local/declarativo de memória e divisão de trabalho, sem nuvem |
| `autosynth_cli.py` | CLI para `profile`, `search`, `verify`, `rollback` e `capacity` |

## Por que AST e não texto aleatório?

Todas as folhas da árvore sintática são construídas com a API `ast` do Python. Cada programa candidato é **sintaticamente válido por construção**. A gramática tipada impede candidatos que invoquem módulos arbitrários, escrevam arquivos ou manipulem credenciais. **Correção da função** é questão diferente da correção sintática: configurações que quebram empates, limites de `k` ou ordenação são rejeitadas.

O espaço demonstrativo tem apenas 4 × 3 × 2 = **24 programas**. A busca realiza 70–96 simulações e reusa cache. Não há bilhões de candidatos, nem alegação de busca universal. Para expandir o domínio seria necessário desenhar operadores tipados, testes e verificadores específicos de novos componentes (por exemplo kernels matriciais com tensor shapes e gradientes).

## Por que `heapq` pode melhorar?

Quando o número de destinatários `k` é muito menor que `n` candidatos, ordenar tudo custa normalmente `O(n log n)`, enquanto selecionar os `k` melhores com heap pode custar `O(n log k)`. É uma transformação clássica, não uma invenção de arquitetura neural. O sistema a encontrou dentro da gramática disponibilizada; **ele não inventou `heapq`**.

## Provas e limites

O gate inspeciona ASTs canônicas, executa verificações diferenciais para vetores de tamanho 0..5 com pontuações em `{-2,-1,0,1}`, incluindo `k` inválido, e confirma que os laços visíveis têm limites finitos. Os métodos `sorted`, `heapq.nsmallest` e `min` seguem especificações da biblioteca padrão, mas a implementação não produz teoremas Lean/Coq comprovando equivalência para todo vetor real possível. `machine_checked_universal_proof = false` é um campo do relatório.

A avaliação também utiliza casos com empates, entradas vazias, valores negativos e tamanhos maiores que os da busca. Fontes fornecidas externamente são rejeitadas se não corresponderem exatamente ao AST gerado pela DSL. O processo filho usa orçamento de tempo, CPU, memória virtual e tamanho de arquivo **onde o sistema operacional suporta**. Isto é defesa em profundidade, **não um isolamento contra código malicioso arbitrário**.

## Autoaperfeiçoamento: ciclo real

- Com consentimento explícito, a versão superior substitui o plugin em `managed/ast_router.py` e a função na RAM é atualizada.
- A versão anterior é copiada para `managed/ast_router.previous.py`.
- A próxima rodada faz o benchmark **contra a versão já melhorada**, não contra um baseline antigo.
- Se a melhoria incremental não superar 5% em exemplos inéditos, o programa não é promovido.
- `rollback --approve` restaura a versão anterior.
- Arquivos centrais, guarda de alinhamento, credenciais e APIs de nuvem **não** são editados/provisionados autonomamente.

## Integração com o laboratório anterior

O projeto contém também os módulos antigos do NEXUS-M: sociedade neural com 100 redes, busca MCTS em uma gramática de operações inteiras, memória SQLite, causalidade, observabilidade, precisão decimal e arquitetura dinâmica. A nova experiência otimiza **um componente real de roteamento**, e não substitui todos os módulos anteriores.

## Comandos

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m nexus_m.autosynth_cli profile --output runs/autosynth
python -m nexus_m.autosynth_cli verify --strategy heap
python -m nexus_m.autosynth_cli search --simulations 96 --output runs/autosynth
python -m nexus_m.autosynth_cli search --simulations 96 --output runs/autosynth --approve
python -m nexus_m.autosynth_cli rollback --output runs/autosynth --approve
python -m nexus_m.autosynth_cli capacity --agents 100 --width 10000 --nodes 100000 --simulations 100000
```

Para testar a segunda iteração, repita `search` no mesmo `--output`. O roteador ativo da primeira rodada torna-se a nova referência.
