# Experimentos anteriores preservados

`simulacoes_iniciais/`: simulação de comunicação 100x100 por 10 ciclos;
detecção de falhas correlacionadas por 1000 ciclos, arquivos de dados,
figuras e checkpoint (PyTorch). Não há MCTS nesses modelos.

`benchmark_neuronal_anterior/`: escala 100, 1000 e 10000 unidades por rede
para detecção estatística em sensores industriais sintéticos, com código,
figuras, CSV, pesos e ZIP históricos.

Os modelos históricos usam **problema e arquiteturas diferentes** do
NEXUS-M novo, que emprega busca simbólica. Comparar diretamente suas taxas
de acerto com o benchmark atual não é válido.
