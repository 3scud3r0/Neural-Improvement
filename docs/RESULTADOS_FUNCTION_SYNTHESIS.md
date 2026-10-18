# Experimentos com funções gerais

Executados em uma cópia descartável do projeto, Linux, Python 3.12.14 e
PyTorch 2.14.1+cpu. Os valores abaixo são medições reais dos contratos dos
exemplos, não os tempos controlados usados nos testes de promoção.

| Alvo | Implementação selecionada | Original, validação final | Candidata, validação final | Redução medida |
|---|---|---:|---:|---:|
| `SharedMCTS._select` em `nexus_m/mcts.py` | Loop com estatísticas de `SearchNode` inline | 74,586 ms | 50,616 ms | 32,1% |
| `attention` em `examples/function_synthesis/attention.py` | `scaled_dot_product_attention` | 63,964 ms | 39,358 ms | 38,5% |

MCTS: tempo de CPU para 5.000 seleções com 64 filhos. Atenção: tempo de parede
para 120 forwards com shape `(2, 4, 128, 32)`, em um thread CPU. Cada número é
a mediana de três processos de benchmark, cada um tomando a mediana de cinco
amostras internas, após aquecimento. Não são latências de uma busca ou de um
modelo inteiro e não devem ser extrapolados para CUDA ou outros workloads.

Os candidatos foram promovidos automaticamente apenas na cópia de teste. Os
contratos verificaram a seleção e integração do MCTS, além de forward e
gradientes da atenção em casos independentes. O rollback de ambos restaurou
os bytes originais dos arquivos, conferidos contra a árvore de trabalho.

Na exploração anterior, propostas menores para o MCTS passaram pelos testes,
mas não sustentaram o ganho mínimo de 5% na validação final. Foram rejeitadas.
O experimento final avaliou três alternativas e selecionou
`proposals_mcts/inline_statistics.py`. O MCTS recebeu essas alternativas
completas; os resultados **não demonstram invenção autônoma de algoritmos**.
Os testes não estabelecem equivalência universal fora dos contratos fornecidos.

Hashes, ambiente e valores sem arredondamento estão em
[RESULTADOS_FUNCTION_SYNTHESIS.json](RESULTADOS_FUNCTION_SYNTHESIS.json).
Para reproduzir em seu hardware, siga o
[guia de síntese genérica](GENERAL_FUNCTION_SYNTHESIS.md). Os ganhos podem mudar
ou desaparecer; a promoção depende da medição na execução atual.
