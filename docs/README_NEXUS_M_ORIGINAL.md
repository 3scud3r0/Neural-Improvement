# NEXUS-M — Laboratório de inteligência coletiva neural

**Protótipo científico executável. Não constitui AGI, prova de inteligência emergente, nem garantia de superar sistemas convencionais.**

100 redes neurais pequenas/médias/grandes + roteamento aprendido e esparso +
memória persistente + MCTS compartilhado + verificação simbólica + observabilidade
online + precisão numérica adaptativa e backend de precisão decimal arbitrária.

## Início rápido

Requer Python 3.10+; validado com Python 3.11, PyTorch 2.10 CPU.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m nexus_m demo --output runs/demo
python -m nexus_m decimal-demo --digits 100 --output runs/decimal_100.json
```

Executar `python -m ...` dentro da pasta que contém `nexus_m/` também funciona
sem instalar o pacote. Não use checkpoints `.pt` provenientes de fontes não
confiáveis.

## Três escalas — 100 redes, 1.000 passos

```bash
python -m nexus_m train --agents 100 --width 100 --steps 1000 --output runs/100x100
python -m nexus_m train --agents 100 --width 1000 --steps 1000 --output runs/100x1000
python -m nexus_m train --agents 100 --width 10000 --steps 1000 --output runs/100x10000
```

`--width` é o número **real de ativações/neurônios por agente**. As conexões são
fatoradas/compartilhadas e cada agente tem parâmetros adaptativos próprios.
Isso não equivale a usar 100 matrizes densas 10000x10000.

Presets JSON reproduzíveis, com buscas periódicas opcionais:

```bash
python scripts/run_config.py configs/100_redes_100_neuronios_1000_ciclos.json
python scripts/run_config.py configs/benchmark_tres_escalas.json
```

**Atenção:** o preset completo de benchmark faz três sementes, 1000 MCTS por
problema, até 128 níveis, várias ablações e todas as escalas. É MUITO MAIS
caro que executar 1000 ciclos de treinamento. Pode demandar hardware robusto.

## Buscar programas com MCTS

```bash
python -m nexus_m solve \
  --model runs/100x100/model_final.pt \
  --simulations 2000 --depth 128 \
  --family shift_absolute --examples 5 \
  --memory runs/100x100/memory.sqlite --output runs/search
```

O problema padrão é **síntese de programas de inteiros a partir de cinco pares
entrada/saída**. O MCTS escolhe tokens de uma gramática RPN segura (não usa
`eval` nem `exec`); programas candidatos são testados com aritmética inteira
exata dentro de limites explícitos. A busca NÃO acessa exemplos de avaliação
nem o programa gerador da tarefa. Uma solução correta em cinco exemplos pode
falhar em dados inéditos: o relatório separa essas duas métricas.

`max_depth=128` é um limite máximo permitido, não promessa de que um milhão de
simulações alcançará profundidade 128. A altura efetivamente atingida é medida.

## Modelo de mundo limitado (opcional)

```bash
python -m nexus_m world-train --steps 1000 --output runs/world
python -m nexus_m solve --model runs/100x100/model_final.pt \
  --world-model runs/world/world_model.pt --world-weight 0.5 \
  --simulations 1000 --depth 40
```

O preditor aprende transições **de estados de programas na DSL**, não leis
universais de física. A busca continua usando o verificador exato.

## Benchmark com ablações

```bash
python -m nexus_m benchmark \
  --widths 100 1000 10000 --agents 100 \
  --steps 1000 --simulations 500 --episodes 30 \
  --seeds 2026 2027 2028 \
  --depth 128 --output runs/benchmark
python scripts/resumir_experimento.py runs/benchmark
```

Quatro condições: neural coletivo com memória; neural coletivo sem memória;
neural isolado; MCTS com política uniforme (sem redes). Mesmos dados de teste,
simulações e limite de profundidade. Os dados e relatórios incluem o custo de
treinamento e quantas simulações foram realmente executadas; mais largura
significa custo computacional maior mesmo para número igual de passos.

O script `scripts/avaliar_checkpoints.py` compara diretamente os checkpoints
já treinados, sem repetir o treinamento.

## Telemetria em tempo real

```bash
python -m nexus_m train --agents 100 --width 100 --steps 20 \
  --precision float64 --telemetry full --log-every 1 \
  --output runs/observability
python scripts/monitorar_eventos.py runs/observability/telemetry/events.jsonl
```

- `off`: mínimo custo, sem log de telemetria.
- `summary`: estados RMS, mensagens ativas, entropia de influência, normas e
  diferenças dos parâmetros e gradientes, eventos de busca.
- `full`: **vetores completos de estados, mensagens, roteamento, pesos** a cada
  passo registrado, em arquivos `.pt`, e índice `.jsonl` legível incrementalmente.

Com 100 redes x 10000 neurônios x 1000 passos, só salvar ativações em FP32
representa ~4 GB, fora mensagens, checkpoints e estados do otimizador.
`full` é intencionalmente opcional por causa de disco/I/O.

## Precisão numérica

- `float32`: pesos e mensagens FP32.
- `float64`: pesos e mensagens FP64 (CPU pode ser bem mais lento).
- `adaptive`: pesos e mensagens FP32; MCTS acumula estatísticas em `float64`
  Python; executores DSL usam inteiros exatos; verificações Decimal disponíveis.
- `decimal-demo --digits 100`: usa **pesos, mensagens e SGD reais com `Decimal`**
  com 100 dígitos significativos em um **micro-modelo CPU lento**. Não afirma
  treinamento de um milhão de neurônios com 100 dígitos, nem ganhos universais.

Mais casas decimais não criam informação, especialmente quando o treinamento
é estocástico ou o problema é mal especificado.

## Arquivos

- `nexus_m/model.py`: população neural, roteamento top-k e inferência recorrente.
- `nexus_m/world_model.py`: modelo preditivo opcional de transições de programas.
- `nexus_m/grammar.py`: linguagem segura, operações e verificador exato.
- `nexus_m/mcts.py`: MCTS PUCT compartilhado, rollouts e estatísticas.
- `nexus_m/memory.py`: armazenamento episódico, conceitos e revalidação.
- `nexus_m/learning.py`: 1000 ciclos, imitação sintética, distilação da busca,
  checkpoint, calibração de adaptadores.
- `nexus_m/decimal_network.py`: protótipo realmente arbitrário de precisão.
- `nexus_m/telemetry.py`: eventos em tempo real, pesos, mensagens, snapshots.
- `nexus_m/diagnostics.py`: intervenções de comunicação e lesão de agentes.
- `nexus_m/benchmark.py`: avaliação fora da família e controles.
- `configs/`: presets de execução.
- `scripts/`: execução por config, monitoramento, resumo, comparação.
- `tests/`: testes determinísticos de integração, memória, segurança etc.
- `legacy/`: experimentos históricos de correlação industrial e comparação de largura.
- `outputs/`: checkpoints realmente treinados e métricas medidas desta execução.

Leia `docs/ARQUITETURA.md`, `docs/METODOLOGIA.md` e `docs/LIMITACOES.md`.

## Problema próprio com pares de exemplos

Edite `examples/problema_usuario.json` ou crie um JSON semelhante com
`examples` e, opcionalmente, `heldout`. A busca NUNCA acessa `heldout`;
estes pares só são usados no relatório de avaliação depois da busca.

```bash
python -m nexus_m solve --examples-file examples/problema_usuario.json \
  --model runs/100x100/model_final.pt --simulations 10000 --depth 40
```

A linguagem aceita inteiros limitados e as instruções implementadas em
`nexus_m/grammar.py`. Não consegue sintetizar automaticamente código Python
arbitrário ou programas fora desse espaço de busca.

## Adaptação durante inferência

```bash
python -m nexus_m solve --model runs/100x100/model_final.pt \
  --adapt-steps 5 --simulations 2000 --output runs/solve_adaptativo
```

Se o MCTS encontrar um programa compatível com todos os exemplos públicos,
o sistema adapta temporariamente ganhos, vieses e roteamento por gradiente,
repete a busca e restaura o checkpoint anterior. O alvo é a solução obtida
pelo próprio MCTS, que pode estar sobreajustada: isso não é prova de melhora.
