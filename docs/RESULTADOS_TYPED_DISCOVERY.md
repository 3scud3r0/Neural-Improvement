# Resultados da descoberta tipada e VM

Ambiente: Linux, Python 3.12.14, PyTorch 2.14.1+cpu, sem CUDA e sem Triton.
Z3 foi instalado e seus testes foram executados. Valores completos e hashes
estão em [RESULTADOS_TYPED_DISCOVERY.json](RESULTADOS_TYPED_DISCOVERY.json).

## Síntese automática em cópia descartável

Não foram fornecidos arquivos `proposals` nem contratos de teste escritos
manualmente nesses experimentos. As funções originais forneceram a
especificação; os drivers e casos foram gerados automaticamente. E-graph e
enumeração tipada produziram as alternativas. Toda proposta encaminhada tinha
prova no fragmento polinomial de inteiros; o tempo foi medido em processos
separados usando a mediana de repetições.

| Experimento | Resultado real | Ganho no benchmark gerado |
|---|---|---:|
| `redundant_identity`, um arquivo | Promovido; simplificado para `x` | 73,9% |
| Primeiro bundle: `expand_square` e `redundant_double` | Rejeitado por falta de ganho, apesar de correção | -8,5% |
| Bundle: `redundant_identity` e `redundant_double`, dois módulos | Validado e aplicado com journal | 62,2% |

Os exemplos contêm redundâncias deliberadas. Esses números medem a remoção
dessas redundâncias sob os inputs/benchmark gerados; não demonstram criação
de novos algoritmos de ordenação, atenção ou inteligência geral. As referências
e workloads dos dois bundles são diferentes; os percentuais não são uma
comparação direta entre duas versões do mesmo experimento.

O experimento individual foi revertido antes do bundle final. Depois, a
recuperação do bundle restaurou os bytes originais de ambos os módulos,
conferidos contra os arquivos de origem. A árvore principal do repositório
não foi auto-modificada por esses experimentos.

## VM em CPU

Comando medido:

```bash
python scripts/benchmark_typed_vm.py --device cpu --programs 512 --inputs 128 \
  --threads 1 --repeats 3 --output runs/vm_cpu.json
```

O lote repete quatro programas distintos, totalizando 65.536 pares
programa × entrada e 327.680 slots de instrução. Os 128 sobreviventes da VM
coincidiram com os verificadores escalares. A comparação usou 2 chunks, com
estimativa de 14.680.064 bytes de working set máximo por chunk.

| Métrica | Medição |
|---|---:|
| Compilação da matriz | 1,835 ms |
| VM vetorizada, mediana | 29,645 ms |
| Pares programa × entrada por segundo | 2.210.727 |
| Escalar, percorrendo todos os pares | 27,076 ms |
| Escalar com interrupção na primeira falha | 8,734 ms |

Neste lote em CPU, a VM ficou **mais lenta** que o escalar, sobretudo quando
o escalar pôde parar cedo. A implementação faz checagens e operações tensoriais
que não desaparecem por usar PyTorch. Este resultado não deve ser apresentado
como aceleração GPU, como `10^9` avaliações/s, nem como latência de ASTs Python
arbitrárias. O backend escalar continua sendo o padrão da busca automática.

## Verificação automatizada

`python -m pytest -q tests examples/function_synthesis`: **114 passaram, 2
ignorados** por ausência de CUDA/Triton. Os testes incluem:

- Congruência e extração cíclica de e-graphs; recusa de floats e nós desconhecidos.
- Refutação, igualdade polinomial, definedness e divisão negativa em Z3.
- Busca que retoma candidatos e respeita combustível; prefiltro vetorizado.
- Todos os opcodes comparados com um interpretador escalar independente,
  incluindo extremos int64, NOPs, divisão por zero, overflow e pilha inválida.
- Dependências por imports relativos, aliases, métodos e chamadas transitivas.
- Bundle correto, bundle incorreto, falha durante a segunda instalação,
  processo encerrado abruptamente e recuperação sem destruir edições externas.
- Oráculos gerados, promoção automática, modo de inspeção e rollback.

As decisões de promoção dos testes usam tempos controlados para evitar
dependência da carga do host. Os números experimentais acima são medições
reais, separadas desses tempos controlados. Os testes condicionais de CUDA e
Triton ainda exigem uma execução real nesses backends.
