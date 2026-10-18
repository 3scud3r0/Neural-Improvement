# Resultados realmente medidos no pacote

## Treinamentos concluídos: 100 redes, 1000 ciclos cada

| Neurônios/rede | Total de ativações | Parâmetros | Duração aprox. CPU | Última perda de treino |
|---:|---:|---:|---:|---:|
| 100 | 10.000 | 42.350 | 3,95 s | -0,0118 |
| 1.000 | 100.000 | 404.150 | 8,33 s | -0,0126 |
| 10.000 | 1.000.000 | 4.022.150 | 36,79 s | 0,0254 |

A perda pode ficar ligeiramente negativa devido ao termo de regularização
por entropia dos pesos de influência; não é uma acurácia e nem garantia de
solução. Os três modelos possuem checkpoints `model_final.pt` em `outputs/`.
Os tempos referem-se somente à execução nesta CPU, e não são estimativas
universais.

## Primeira avaliação OOD

Oito tarefas inéditas, cinco exemplos públicos por tarefa, 20 entradas
ocultas, 150 simulações MCTS, profundidade máxima 25, semente 2027.

| Neurônios por agente | Política | Acurácia nos 160 pares ocultos | Tarefas integralmente corretas |
|---:|---|---:|---:|
| 100 | Coletiva + memória | 18,125% | 0/8 |
| 100 | Coletiva sem memória | 18,125% | 0/8 |
| 100 | Isolada (ablação) | 15,625% | 0/8 |
| 100 | MCTS uniforme | 6,875% | 0/8 |
| 1.000 | Coletiva + memória | 19,375% | 0/8 |
| 1.000 | Coletiva sem memória | 19,375% | 0/8 |
| 1.000 | Isolada (ablação) | 24,375% | 1/8 |
| 1.000 | MCTS uniforme | 6,875% | 0/8 |
| 10.000 | Coletiva + memória | 23,125% | 0/8 |
| 10.000 | Coletiva sem memória | 23,125% | 0/8 |
| 10.000 | Isolada (ablação) | 0% | 0/8 |
| 10.000 | MCTS uniforme | 6,875% | 0/8 |

Esses resultados são **fracos** em generalização e não demonstram
superioridade de uma arquitetura. O ganho em pares individuais não garante
resolver o programa inteiro. A memória não ajudou neste conjunto.
Apenas uma semente de treinamento e oito tarefas são insuficientes
para conclusões estatísticas. Consulte `outputs/comparacao_treinados/`.

## Outros testes de execução

- Testes automatizados de gramática segura, memória persistente, MCTS,
  intervenções, Decimal, FP64, modelo preditivo e pipeline integrado.
- Experimento `outputs/decimal_100_digitos.json`: microrede com pesos e
  mensagens em `Decimal` de 100 dígitos, atualizados por SGD manual.
- Experimento `outputs/world_1000/`: 1000 passos de aprendizado de transição
  de representações da DSL, não física real.
- Experimento `outputs/telemetria_fp64_demo/`: 10 passos de rede 100x100 em
  FP64 com salvamento integral de ativações, mensagens e pesos.

## Interpretação

Os componentes estão implementados e interagem, mas a eficácia do conjunto
é uma hipótese aberta. Há gargalos de exploração combinatória, programas
espúrios que passam poucos exemplos e generalização ruim. O próximo avanço
exigiria busca estruturada com propostas hierárquicas, feedback mais útil e
mais repetições sob orçamento de computação controlado.
