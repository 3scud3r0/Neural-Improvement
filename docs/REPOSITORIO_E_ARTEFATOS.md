# Repositório e artefatos

Este repositório reúne o código-fonte executável da última evolução NEXUS AutoSynth, incorporando o NEXUS Evolution Lab e o NEXUS-M original.

## Conteúdo versionado

- `nexus_m/`: arquitetura neural, MCTS, síntese AST, causalidade, memória, autoaperfeiçoamento com aprovação e rollback.
- `tests/`: testes automatizados (45 aprovados na execução de entrega).
- `configs/`, `docs/`, `formal/`, `examples/` e `scripts/`: configurações, documentação científica, protocolo de verificação e exemplos.
- `legacy/`: código dos experimentos iniciais e do benchmark neuronal.
- `outputs/`: resumos textuais de benchmarks medidos.
- `NEXUS_AUTOSYNTH_CODIGO_TOTAL.txt`: concatenado auditável do código e da documentação (não executável isoladamente).

## Artefatos grandes excluídos

Os arquivos binários de checkpoint (.pt), logs SQLite, gráficos PNG e os arquivos ZIP grandes dos lançamentos anteriores **não estão hospedados neste repositório**. O arquivo ZIP completo da entrega anterior tem cerca de 183 MB e alguns checkpoints individuais chegam a dezenas de megabytes. Para versioná-los recomenda-se Git LFS ou publicá-los como ativos de uma GitHub Release, com checksums.

## Teste de integridade

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m nexus_m.autosynth_cli search --simulations 96 --output runs/autosynth
```

Os resultados do relatório não implicam AGI ou autoaperfeiçoamento irrestrito. A verificação finita não equivale a uma prova universal.
