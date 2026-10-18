# Experimentos realmente executados

Executados neste ambiente com Python 3.13 e CPU, sem dados externos.

## Demonstração integrada

Comando:

```bash
python -m nexus_m.evolution_cli demo \
  --output outputs/evolution_demo_v2 --simulations 500
```

- Interpretador exato computou `6! = 720`.
- SymPy provou `(x+1)^2 = x^2+2*x+1` na gramática suportada.
- Ajuste frontdoor recuperou a intervenção `P(Y=1|do(X=1)) = 0,887` até erro
  numérico de `2,22e-16` em um SCM binário conhecido.
- Modelo físico sintético restrito: R² reservado `0,9999625`.
- Busca MCTS inicial encontrou `f(x)=2*x+3` com 89 simulações na execução
  sem treinamento do novo prior; política uniforme encontrou solução em 45.
- Reuso de memória recuperou a solução sem reiniciar a busca.
- Destilação de 8 passos reduziu a perda de aproximadamente 2,95 para 2,69.
- Modelo latente de 80 passos reduziu a perda numérica de aproximadamente
  0,104 para 0,002. A similaridade cosseno reservada foi ~0,990 versus
  ~0,956 no controle com alvos embaralhados; isso não isola generalização.
- MCTS com prior latente encontrou um programa consistente após 11 simulações
  nessa execução (não constitui média sobre problemas).
- Uma proposta de roteamento heap foi ~58% mais rápida do que o baseline
  sort naquele microbenchmark, mas **não foi aplicada** por falta de autorização.

Todos os resultados completos, inclusive os caminhos e diferenças de tempo,
estão em `outputs/evolution_demo_v2/demo_result.json`.

## Benchmark de 24 tarefas

Comando:

```bash
python -m scripts.benchmark_evolution --seeds 2026 2027 2028 \
   --simulations 350 --output outputs/evolution_benchmark
```

| Abordagem | Teste exato | Simulações médias | Tempo apenas da busca |
|---|---:|---:|---:|
| Uniforme | 20/24 | 146,2 | 0,30 s agregado |
| Neural não treinada | 15/24 | 198,1 | 0,68 s agregado |
| Latente | 23/24 | 80,0 | 5,15 s agregado |
| Coletiva online | 20/24 | 141,3 | 10,44 s agregado |

**As simulações são uma medida incompleta de custo.** Priors latentes são
mais caros para calcular, e o treinamento de latentes/destilação não entra no
tempo de busca da tabela. Por isso, a configuração latente usa menos
simulações, mas não necessariamente menos tempo total. As métricas foram
computadas em 24 problemas provenientes de 8 modelos algorítmicos (3 sementes),
todos expressáveis pela mesma gramática predefinida; não representam AGI.

## Limitações e ameaças à validade

- Poucos problemas, poucos seeds; variação estatística elevada.
- Similaridade estrutural entre os algoritmos do domínio e ações permitidas.
- O mesmo conjunto de operações está disponível na busca e na geração dos
  problemas; não é generalização universal entre domínios.
- Atualização online favorece uma condição com computação adicional.
- O treinamento latente usa transições sintéticas da gramática dos testes,
  sem rótulos dos problemas de teste; portanto não é zero conhecimento prévio.
- Memória recupera apenas programas verificados nos exemplos disponíveis,
  sem garantia de validade em outras entradas.

Arquivos brutos: `outputs/evolution_benchmark/results.csv` e `summary.json`.
