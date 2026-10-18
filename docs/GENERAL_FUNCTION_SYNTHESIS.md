# MCTS sobre funções Python do repositório

Veja também os [resultados medidos em funções de MCTS e atenção](RESULTADOS_FUNCTION_SYNTHESIS.md).

`function-search` remove a dependência de `RouterSpec` e do contrato
`route(scores, k)`. O alvo é um arquivo Python e o nome qualificado de uma
função. A busca opera sobre ASTs completas de funções, avalia alterações em
cópias do projeto e promove a fonte quando testes e desempenho permitem.

```bash
python -m nexus_m.autosynth_cli functions --target nexus_m/mcts.py
python -m nexus_m.autosynth_cli function-search \
  --workspace . --target nexus_m/mcts.py --function SharedMCTS._select \
  --contract examples/function_synthesis/mcts_contract.json \
  --proposals examples/function_synthesis/proposals_mcts \
  --no-mutations --simulations 2 --depth 1 \
  --output runs/function_mcts
```

O exemplo acima **pode editar `nexus_m/mcts.py`**. Use uma cópia do projeto para
experimentar sem alterar sua árvore principal. `--no-approve` executa busca e
validação, mas deixa a proposta preparada para inspeção. `--min-gain` controla
o ganho mínimo (padrão 5%). Candidatos corretos podem ser rejeitados por falta
de ganho. O comando não garante descobrir uma implementação melhor.

## Alvos e escopo da edição

- Funções de módulo: `evaluate`, `attention`.
- Métodos e propriedades: `SharedMCTS._select`, `Attention.forward`.
- Funções internas: `outer.inner`; classes internas também podem compor o nome.
- Funções assíncronas, decoradas e definições dentro de condicionais.

O patch substitui o intervalo da função escolhida, inclusive seus decorators.
Comentários e bytes fora desse intervalo são preservados, assim como a
codificação e as quebras de linha do arquivo. Definições com o mesmo nome no
mesmo escopo são consideradas ambíguas e precisam ser desambiguadas na fonte.
Wrappers compilados sem fonte Python não têm AST Python selecionável.

Uma proposta pode alterar corpo, argumentos, defaults, decorators, anotações,
imports internos, helpers internos e até a natureza síncrona/assíncrona. Cabe
ao contrato verificar se a interface continua compatível. O nome da função
deve continuar igual ao do alvo. Para adicionar imports ou helpers, coloque-os
dentro da função ou disponibilize-os no projeto; este modo não aplica patches
independentes a outros pontos do módulo na mesma proposta.

## Propostas e busca

Há três fontes de ações:

1. Mutações de AST independentes do domínio: redução de loop para `sum`,
   comprehension para generator, operadores, constantes e retorno condicional.
2. `--proposals arquivo.py` ou diretório: implementações completas alternativas.
   Cada arquivo contém uma função inteira, incluindo seus decorators quando
   necessários. Não existe allowlist de nós AST nem assinatura de roteador.
3. `--provider pacote.modulo:callable`: plugin de busca confiável que recebe
   `(function_ast, random.Random)` e retorna objetos
   `nexus_m.function_ast.Proposal(label, function_ast)`.

As opções `--proposals` e `--provider` podem ser repetidas. Propostas externas
são exploradas antes das mutações nativas. `--no-mutations` desliga apenas as
mutações nativas. Plugins executam no processo de busca e devem ser código
confiável; podem conectar outros geradores, um corpus ou modelos de código.
Um exemplo mínimo de plugin funcional:

```python
from copy import deepcopy
from nexus_m.function_ast import Proposal

def propose(function_ast, rng):
    alternative = deepcopy(function_ast)
    # Gerador de exemplo: retira apenas o docstring, quando presente.
    import ast
    first = alternative.body[0]
    if (len(alternative.body) > 1 and isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)):
        alternative.body = alternative.body[1:]
        yield Proposal("remove-docstring", alternative)
```

Remover um docstring muda comportamento observável; o contrato pode rejeitar
essa proposta. O exemplo demonstra a interface, não uma otimização garantida.

Cada nó do MCTS guarda uma função completa. As arestas são propostas, podendo
compor alterações ao longo da árvore. A seleção usa UCB; o retorno combina
sucesso dos testes e ganho medido. Falhas recebem recompensa negativa. Há cache
por hash do módulo candidato, rastreio das decisões e rejeição de ciclos de
AST no caminho atual. `--depth`, `--max-children` e `--simulations` definem o
orçamento de busca; não há catálogo fixo de 24 programas neste modo.

As mutações nativas são um ponto de partida pequeno. Aceitar ASTs de qualquer
função **não equivale a conseguir inventar qualquer algoritmo**: a descoberta
depende dos provedores, da informação disponível e do orçamento. O exemplo
de atenção fornece explicitamente o algoritmo alternativo ao MCTS; ele não
foi inventado autonomamente durante o experimento.

## Contrato de avaliação

O contrato é JSON, com comandos como arrays de argumentos, sem shell:

```json
{
  "tests": [["{python}", "-m", "pytest", "tests/test_target.py", "-k", "training"]],
  "heldout": [["{python}", "-m", "pytest", "tests/test_target.py", "-k", "heldout"]],
  "benchmark": ["{python}", "bench_target.py"],
  "repetitions": 3,
  "timeout_seconds": 60,
  "benchmark_timeout_seconds": 60,
  "seed": 2026
}
```

`{python}` usa o interpretador ativo. Cada candidato é aplicado a uma cópia
nova do projeto, que mantém os imports relativos e usa as dependências
instaladas nesse interpretador. Git, caches, ambientes virtuais, `runs`, o
diretório de artefatos e nomes comuns de credenciais não são copiados. Symlinks
são recusados. O alvo precisa estar incluído na cópia.

As validações rodam em processos separados, com cwd e `PYTHONPATH` apontando
para a cópia. Variáveis comuns de credenciais são removidas. HOME e pacotes de
usuário permanecem disponíveis para compatibilidade com o ambiente ativo.
Timeout encerra o grupo POSIX ou a árvore Windows. Esse avaliador **não é um
sandbox de kernel**, não bloqueia rede nem acesso absoluto ao host; para esse
isolamento, execute o laboratório em VM ou em um sandbox externo adequado.

O benchmark deve gravar `{"seconds": numero_positivo_finito}` no arquivo
indicado por `NEXUS_BENCHMARK_FILE`. O objetivo é minimizar esse valor. Meça o
trabalho alvo dentro do processo, com aquecimento e sincronização CUDA quando
necessários, para excluir tempo de imports. O avaliador usa a mediana das
repetições; não presume que tempo de execução de testes seja desempenho da
função. `NEXUS_EVALUATION_SEED` diferencia busca e validação final.

Antes da busca, a fonte original deve passar no contrato de busca. Para
promover, há uma nova avaliação da original e da candidata, incluindo os
comandos `heldout`, com novo benchmark. O ganho mínimo precisa passar de novo.
Se a melhor candidata falhar nessa validação, as próximas elegíveis são
avaliadas. Sem `heldout`, o resultado é apenas preparado, sem promoção.

Os testes são a especificação disponível, não prova de equivalência universal.
Para atenção, inclua gradientes, precisões, shapes, masks, dropout e dispositivos
pertinentes ao seu uso. Para MCTS, inclua critérios de empate, estados vazios,
resultados da busca e efeitos de sementes. Arquivos de teste presentes na cópia
não são um conjunto secreto protegido contra candidatos hostis. Contratos e
métricas podem ser insuficientes ou manipulados por código com essas permissões.

## Exemplo de atenção em PyTorch

```bash
python -m nexus_m.autosynth_cli function-search \
  --workspace . --target examples/function_synthesis/attention.py \
  --function attention \
  --contract examples/function_synthesis/attention_contract.json \
  --proposals examples/function_synthesis/proposals_attention \
  --no-mutations --simulations 1 --depth 1 \
  --output runs/function_attention
```

A proposta troca atenção explícita por
`torch.nn.functional.scaled_dot_product_attention`. O contrato verifica forward,
shapes distintos, entradas não contíguas e gradientes float64. Esse alvo não
tem masks/dropout em sua assinatura; o contrato não certifica essas modalidades
nem comportamento geral em CUDA. O benchmark mede atenção CPU em um thread.
O benchmark de MCTS mede tempo de CPU do processo para reduzir ruído de
agendamento; os valores não representam latência de uma aplicação inteira.

Para editar uma função Python do próprio PyTorch, use um checkout dele como
`--workspace`, o arquivo relativo como `--target` e um contrato adequado àquele
build. As dependências e extensões compiladas precisam estar preparadas para
que o import realmente use essa árvore. Alterar Python não reescreve kernels
C++/CUDA do PyTorch; isso exigiria outro backend, compilação e outros testes.

## Artefatos, promoção e rollback

O diretório `--output` contém o módulo candidato, a função, o diff, candidatos
avaliados por hash, o trace do MCTS e `report.json`, com resultados de comandos,
medidas, ganho, hashes e escopo de verificação. As três propostas principais
são `candidate_module.py`, `candidate_function.py` e `candidate.patch`.
Uma nova execução limpa propostas selecionadas antigas para evitar confusão.

Promoção segue a política automática existente, grava o módulo original como
backup e substitui a fonte por rename atômico, preservando suas permissões.
O alvo é comparado com o hash de origem depois da avaliação e antes da
promoção; uma edição concorrente detectada interrompe a aplicação. O código
novo entra em vigor em **processos novos**, sem trocar funções carregadas pelo
próprio pesquisador durante a avaliação. Isso permite alterar o sintetizador
sem mudar a implementação que está conduzindo a rodada atual.

```bash
python -m nexus_m.autosynth_cli function-rollback \
  --workspace . --output runs/function_mcts
```

O rollback verifica o hash do backup e da fonte promovida e recusa sobrescrever
edições posteriores. Ele restaura o arquivo original inteiro. A autorização
continua limitada à raiz do workspace, abrangendo também módulos centrais.

## Testes

```bash
python -m pytest -q tests examples/function_synthesis
```

Os testes de integração verificam chamadas reais em processos filhos e pacotes
copiados, funções fora do domínio top-k, rejeição de propostas quebradas ou
que só passam na busca, timeout, seleção de métodos/async/nested, preservação
de fonte, composição de edições, rollback e alterações concorrentes. Para
testar decisões de promoção sem depender da carga do host, esses testes usam
valores de tempo controlados; as medições reais pertencem aos exemplos acima.
