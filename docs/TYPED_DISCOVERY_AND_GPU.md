# Descoberta tipada, falsificação, transações e VM CPU/CUDA

Esta etapa conecta as ideias úteis das análises a implementações executáveis,
com domínios e resultados explícitos. Não implementa uma Máquina de Gödel,
não supera o problema da parada e não torna intenção humana inferível a partir
de um benchmark. O backend geral de ASTs do PR anterior continua disponível.

## Instalação e execução

```bash
python -m pip install -e ".[dev,smt]"
python -m pytest -q tests examples/function_synthesis

# Descobrir uma simplificação sem arquivos de propostas nem contrato manual.
# Use uma cópia do checkout para os experimentos.
python -m nexus_m.sovereign_improver improve \
  --workspace . \
  --symbol examples.typed_discovery_demo.arithmetic.redundant_identity \
  --max-fuel 2000 --simulations 2 \
  --output runs/typed_single --no-approve

# Análise estática de dependentes.
python -m nexus_m.sovereign_improver impact \
  --symbol examples.typed_discovery_demo.arithmetic.redundant_identity

# Bundle de duas funções em módulos diferentes.
python -m nexus_m.sovereign_improver improve \
  --symbol examples.typed_discovery_demo.arithmetic.redundant_identity \
  --symbol examples.typed_discovery_demo.secondary.redundant_double \
  --output runs/typed_bundle --max-fuel 2000 --no-approve
```

Sem `--no-approve`, a política padrão permite aplicar uma candidata que atinja
correção e ganho medido. Uma função usa o MCTS genérico; um bundle gera
alternativas provadas para os alvos e valida o conjunto em uma transação. O
bundle não implementa busca MCTS conjunta sobre todos os programas possíveis.
`--contract arquivo.json` permite acrescentar um contrato de projeto explícito;
se omitido, o orquestrador gera o seu contrato de referência.

## O que cada módulo faz

| Módulo | Implementação | Escopo |
|---|---|---|
| `typed_program.py` | IR imutável tipada, geração de termos válidos, interpretação resumível, forma normal de polinômios | Expressões de inteiros, não Python completo |
| `egraph_synthesis.py` | Hash-consing, union-find, reconstrução de congruência, saturação e extração de custo positivo com ciclos | Polinômios puros em inteiros matemáticos |
| `oracle_coevolution.py` | Casos adversariais, comparação de valores/erros, prova por normalização e Z3 opcional | Refutação ou prova dentro da IR; nunca prova de Python arbitrário |
| `levin_universal_search.py` | Dovetailing resumível com cotas por código prefix-free, combustível e orçamento de geração | Busca finita inspirada em Levin, sem teorema de optimalidade universal |
| `whole_program_pdg.py` | Símbolos qualificados, imports relativos, aliases, chamadas de métodos e fechamento de chamadores | Grafo estático de chamadas, não PDG completo |
| `repository_transaction.py` | Validação do bundle, lock cooperativo, backups, journal, instalação e recuperação | Atomicidade por arquivo, não visibilidade global atômica |
| `gpu_program_evaluator.py` | VM de pilha em blocos, aritmética int64 verificada, CPU/CUDA e tempos sincronizados | 12 opcodes, não execução de AST Python na GPU |
| `gpu_falsifier.py` | Referência e candidatas na mesma VM, casos de borda/aleatórios e witnesses | Evidência em lote finito, não equivalência universal |
| `triton_kernel_compiler.py` | Exportação de polinômios afins para callable tensorial Torch ou Triton JIT | Inteiros int64 contíguos; não compilador geral de kernels |
| `sovereign_improver.py` | Orquestração efetiva, referências congeladas, prova, MDL heurístico, benchmark e promoção | Nome histórico; estados só indicam promoção depois de aplicação real |

## Tipagem, axiomas e e-graphs

O modo automático aceita funções síncronas sem decorators, com parâmetros
posicionais simples anotados literalmente `int`, e um corpo com uma expressão
de retorno pura. Essas anotações **declaram** o domínio int; Python não as
impõe em runtime. Não há prova para chamadores que passem floats, tensores,
objetos com operadores sobrecarregados ou outros tipos. Assinaturas mais
complexas e corpos com loops continuam no backend geral, com contratos e
provedores apropriados; não recebem um selo falso de prova automática.

A IR representa constantes, variáveis e operadores inteiros. A enumeração
atual usa `+`, `-`, `*` e quatro constantes iniciais, compondo expressões até
os orçamentos configurados. A VM também implementa divisão inteira, mínimo e
máximo. Chamadas Python não são confundidas com primitivas matemáticas:
`min` e `max` podem estar shadowed no módulo, portanto o extrator automático
de funções não aceita chamadas a eles como axiomas de Python.

O e-graph aplica comutatividade, associatividade, identidades, constant folding,
distributividade e fatoração **somente** a inteiros matemáticos puros. Ele não
aplica comutatividade a matrizes ou floats e não converte nós desconhecidos em
`Raw` ou `Pass`. Não usa `eval()` para recuperar constantes. `rebuild()` unifica
pais congruentes após unions; extração por ponto fixo evita recursão infinita
em classes como `x == x + 0`. As regras recebidas são verificadas e utilizadas.

Custos de extração são um modelo estrutural, não ciclos de CPU medidos. Toda
promoção ainda precisa do benchmark real. A saturação tem orçamentos de nós
e iterações: e-graphs também podem crescer exponencialmente. Não existe aqui
garantia de memória polinomial para qualquer sistema de reescrita.

O provedor também pode ser usado no MCTS geral:

```bash
python -m nexus_m.autosynth_cli function-search \
  --target examples/typed_discovery_demo/arithmetic.py \
  --function redundant_identity --contract meu_contrato.json \
  --provider nexus_m.egraph_synthesis:egraph_proposals --no-mutations
```

## Referência, provas e limites do oráculo

O código existente fornece a especificação inicial. O orquestrador gera
casos de borda e aleatórios, calcula resultados no interpretador da IR e
grava `oracle/reference.json`, com hash embutido no driver externo à cópia.
Busca e validação independente usam sementes distintas. O driver verifica
que o import resolveu para a função no módulo copiado, compara valores e
classes de erro e mede chamadas reais. Ele não aceita `True` como retorno int.
Uma alteração do JSON congelado invalida a execução.

Esse mecanismo reduz dependência de arquivos de asserts escritos para cada
experimento. Ele **não cria a intenção do problema**: preserva a referência,
inclusive seus bugs. Casos novos servem como adversários; não coevoluem
livremente uma nova especificação nem redefinem o objetivo para facilitar a
aprovação. Dados presentes em arquivos não constituem testes secretos contra
código hostil com as permissões do host.

Resultados possíveis do oráculo:

- `proved`: igualdade de polinômios em `Z[variáveis]` por normalização exata,
  ou Z3 retorna `unsat` para desigualdade/definedness da IR.
- `falsified`: há um binding concreto que difere em valor ou erro.
- `tested_no_counterexample`: testes finitos passaram sem prova.
- `unknown`: o resolvedor não concluiu dentro de seu limite.

Sem Z3, casos fora do fragmento polinomial não ganham prova automaticamente.
Com Z3, a tradução modela divisão floor com denominadores negativos e
definedness em zero. Isso ainda não resolve aritmética não linear arbitrária
nem todos os problemas de SMT. Orçamentos de memória/tempo da execução Python
não fazem parte da prova sobre inteiros matemáticos.

O modo automático só encaminha propostas com prova no domínio suportado.
O MCTS pode usar a recompensa que combina ganho medido e comprimento de AST,
mas essa compressão é uma heurística de descrição, **não** complexidade de
Kolmogorov nem uma solução para Goodhart. Código não verificado recebe
recompensa negativa; a heurística não substitui a correção ou o ganho mínimo.

## Busca inspirada em Levin

Programas são gerados como árvores bem tipadas, sem `exec` de candidatos no
processo do pesquisador. Um código prefix-free computável mede o comprimento
de descrição da expressão. Na fase `r`, cada candidato admitido de comprimento
`l` recebe até `2**(r-l)` passos, sob o orçamento global restante. O estado de
verificação é retomado; a busca não recomeça eternamente pelo mesmo prefixo.
Geração de programas tem um orçamento separado e há prazo de parede.

`K(p)` verdadeiro é incomputável. O comprimento desse código é uma descrição
concreta, não seu valor de Kolmogorov. A linguagem atual é finita sob seus
orçamentos e não contém loops livres; o módulo não demonstra optimalidade
universal de Lsearch nem resolve o problema da parada. Mesmo a busca universal
teórica pode pagar constantes impraticáveis e custo exponencial.

`--backend vectorized --device cuda` usa enumeração com **prefiltro em lote**,
não o mesmo escalonamento ponderado. `--max-fuel` passa a limitar slots de
instrução × entradas no prefiltro, em vez de passos do interpretador escalar.
Depois, os sobreviventes são checados pelo escalar e pelo oráculo. O prefiltro
int64 pode perder soluções matemáticas com intermediários fora de int64; ele
não amplia a prova para esse domínio nem é prova por amostragem massiva.
O backend padrão continua escalar; vetorização em CPU pode ser mais lenta.

## Dependências e mudanças coordenadas

Símbolos incluem caminho de módulo e escopo (`pkg.a.Engine.forward`), evitando
colisões entre `a.py` em diretórios distintos. O grafo resolve chamadas locais,
imports relativos e aliases, `self/cls` e alguns objetos construídos localmente.
O relatório inclui chamadas não resolvidas e erros de parsing, com
`complete=false`. Reflection, despacho geral, herança dinâmica, monkeypatching
e imports dinâmicos não são garantidos. Não há uma análise completa de fluxo
de controle/dados para Python.

`create_multi_file_transaction()` renderiza **arquivos completos**, preservando
o contexto; não substitui um arquivo por uma única definição de função. Uma
mudança de argumentos exige incluir os módulos dos chamadores estáticos
atingidos. Essa regra é conservadora e não inventa os adaptadores: as funções
coordenadas ainda precisam ser propostas e os testes precisam demonstrar sua
compatibilidade. O bundle automático atual gera reescritas algébricas que
preservam as assinaturas, não mudanças autônomas de arquitetura neural.

A transação valida todos os arquivos numa mesma cópia do projeto e registra
backups/hashes antes de começar a instalação. Um lock coordena escritores que
usam esse protocolo. Falhas de instalação disparam restauração; interrupção
abrupta deixa journal recuperável:

```bash
python -m nexus_m.sovereign_improver recover \
  --workspace . --output runs/typed_bundle
```

Cada rename é atômico. Leitores externos podem observar um estado intermediário
entre dois renames; `os.replace` não oferece atomicidade global multi-arquivo.
A recuperação verifica todos os hashes antes de restaurar e recusa sobrescrever
uma edição externa desconhecida. A primeira versão substitui arquivos Python
existentes; não cria/exclui módulos. Operadores precisam separar o diretório de
artefatos dos alvos e usar um novo diretório para uma nova transação ainda não
revertida. O lock não é um sandbox e não impede editores que ignorem o protocolo.

## VM vetorizada e memória

```python
import torch
from nexus_m.gpu_program_evaluator import GPUBatchEvaluator

vm = GPUBatchEvaluator(device="auto", max_memory_bytes=128*1024**2)
programs = [["X", "C2", "MUL", "C1", "ADD"], ["X", "X", "MUL"]]
inputs = torch.arange(-100, 101, dtype=torch.int64, device=vm.device)
matrix = vm.compile_bytecode_matrix(programs)
result = vm.evaluate_batch(matrix, 2*inputs+1, inputs)
print(result.surviving_program_indices, result.backend)
```

Compilação não transforma tokens desconhecidos em `NOP` e não trunca programas.
O executor preserva posições de pilha de lanes inativas, detecta underflow e
overflow de pilha, exige um resultado final e distingue divisão por zero,
overflow numérico e erro estrutural. Operações int64 que saem do intervalo
assinado são inválidas, em vez de aceitar wraparound. `MIN_INT // -1` não é
executado como divisão nativa inválida. Referência e candidata passam pelo
mesmo executor, sem uma segunda implementação incompleta de opcodes.

Há duas dimensões de chunking, programas e entradas. A comparação de lotes
acumula os índices globais de sobreviventes e o primeiro contraexemplo de cada
falha, sem alocar a saída completa de todos os programas. `raw_outputs` é
limitado e reserva espaço para suas saídas. A estimativa de working set guia
os blocos; não é uma quota de VRAM do driver nem inclui todos os custos do
allocator e dos tensores fornecidos pelo chamador.

Uma pilha `(8, 65536, 4096)` em int64 exige 16 GiB antes dos temporários, em
memória global. Ela não cabe automaticamente em memória compartilhada/L1.
PyTorch lança operações por instrução/bloco, não um único kernel fundido para
todo programa; a lógica de orquestração ainda usa Python. Operações inteiras
não são automaticamente Tensor Core matmuls.

O falsificador retorna sobreviventes **nesse lote**, com sementes reprodutíveis
e casos extremos. Uma referência estruturalmente inválida é recusada. Ter o
mesmo erro aritmético em ambos os lados pode constituir o mesmo resultado
parcial; isso não é confundido com sucesso numérico ou divergência automática.

Métricas usam sincronização CUDA antes/depois da medição. `total_evaluations`
conta pares programa × entrada; `instruction_slots` conta também comprimento e
padding. Não significa chamadas arbitrárias de Python por segundo. Só o
hardware efetivamente usado recebe medidas.

```bash
python scripts/benchmark_typed_vm.py --device cpu --programs 512 --inputs 128 \
  --threads 1 --output runs/vm_cpu.json
# Em máquina com CUDA configurada, repita explicitamente com --device cuda.
```

O benchmark declara os quatro programas distintos repetidos no lote e mede
verificação escalar completa e escalar com interrupção antecipada separadamente.
O verificador vetorial executa todos os slots, inclusive para candidatas que
já falharam. Não se promete `10^9` avaliações/s ou latência de 5 ms.

## Triton opcional

```bash
python -m pip install -e ".[triton]"
```

`TritonKernelCompiler.status()` declara disponibilidade. O exportador aceita
um polinômio afim de uma variável ou coeficientes explícitos e devolve um
callable para tensores int64 contíguos. CUDA/Triton compila na primeira chamada;
`backend="torch"` oferece referência/fallback rotulado, sem alegar compilação
nativa. Intermediário multiplicativo e resultado precisam caber em int64.
O benchmark deve incluir ou separar o custo dessa validação e do warmup.

Esse callable tensorial é uma interface separada: não substitui silenciosamente
uma função escalar por uma função de tensor. Não compila atenção, classes ou
AST Python arbitrária em kernels CUDA. Sem hardware, os testes CUDA/Triton são
ignorados explicitamente, não considerados aprovados.

## O que foi e não foi superado

A implementação amplia geração automática, evidência e coordenação de arquivos.
Não garante inventar QuickSort a partir de BubbleSort, inferir objetivos do
mundo real, eliminar Goodhart, provar qualquer programa ou evitar toda quebra
de runtime. A linguagem, os axiomas, a especificação e os recursos continuam
fazendo parte do problema. Os limites restantes não são apenas hardware.
