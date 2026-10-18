# Execução local de validação

Este arquivo é atualizado a partir de saídas reais, não de números estimados.

Os códigos foram testados em PyTorch 2.10 CPU. Os checkpoints do diretório
`outputs/` pertencem à versão NEXUS-M desta entrega. Os artefatos anteriores
permanecem em `legacy/`, em tarefas estatísticas diferentes.

Resultados brutos: `outputs/comparacao_treinados/summary.csv` e
`results_by_task.csv`. A geração de resultados foi feita com poucas tarefas
em uma semente para validar o pipeline — insuficiente para inferir vantagem
estatística. **Resultados ruins não são excluídos.**

A avaliação usa famílias de programas não vistas no treino. Cinco exemplos
são usados para sintetizar cada programa. A métrica `ood_exact` só vale 1 se
o programa passar TODOS os testes ocultos. Consulte os dados, não apenas
a perda de treinamento.
