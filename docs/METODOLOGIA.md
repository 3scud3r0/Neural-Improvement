# Metodologia de avaliação

## Tarefa

Síntese de programas de inteiros a partir de pares `(x,y)`. O gerador define
famílias simples para treinamento: deslocamento, afim, quadrado com constante,
valor absoluto, paridade, negação e divisão inteira por dois. Os testes
utilizam **composições de regras inéditas**: absoluto após deslocamento,
quadrática combinada, paridade escalada, produto de deslocamentos etc.

O executor verifica exatamente as operações; os exemplos ocultos usam valores
maiores (`x` fora da faixa típica dos exemplos), fornecendo extrapolação.

## Separação entre avaliação e treinamento

`Task.public()` remove o programa gerador, todos os testes ocultos e o rótulo
da família. O executor MCTS e a memória só recebem `public()`. Somente
`eval_one()` acessa exemplos ocultos depois de a busca terminar. Resultados de
teste não são gravados na memória por padrão. Há testes automatizados disso.

## Variáveis independentes

- Largura: 100, 1000 ou 10000 ativações por agente.
- Comunicação: ativa ou desligada na inferência.
- Memória: recuperação ativa ou desligada.
- Política: neural ou uniforme, sem redes.
- Precisão: FP32/FP64/adaptativa/Decimal microbackend.
- Orçamento de simulações e profundidade de busca.

## Métricas

- `example_exact`: programa coincide com todos os poucos exemplos conhecidos.
- `ood_accuracy`: proporção de pares novos corretamente previstos.
- `ood_exact`: todos os pares ocultos do episódio estão corretos.
- `expanded_nodes`, `simulations`, `max_depth_reached`: gasto e cobertura da busca.
- `elapsed_sec`, `train_sec`: custo medido, não estimado.
- Diversidade e influência: estados internos e peso de cada agente.

Um programa pode ser exato em todos os cinco exemplos e estar errado para
outros valores. Exatidão amostral NÃO equivale a prova de correção global.

## Limitações estatísticas

O número de passos de treinamento igual não significa mesmo custo em FLOPs;
o modelo mais largo costuma ser mais caro. Um experimento rigoroso também
igualaria recursos computacionais e repetiria várias sementes de treinamento.
Vários seeds na avaliação de um mesmo checkpoint medem variabilidade dos
episódios, mas NÃO capturam variação entre treinamentos independentes.

Parâmetros, dados e budgets aqui são escolhas iniciais, não ótimos comprovados.
MCTS com rollout aleatório pode falhar em expressões muito simples ou
produzir programa espúrio quando há apenas cinco exemplos.

## Baseline industrial anterior

`legacy/` contém as simulações anteriores de falhas correlacionadas em
sensores. A tarefa atual de programas foi escolhida para viabilizar MCTS com
verificação exata. Não compare diretamente acurácia da tarefa industrial com
a taxa de síntese de algoritmos: tarefas e métricas são diferentes.
