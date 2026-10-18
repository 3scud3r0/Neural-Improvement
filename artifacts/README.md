# Artefatos e checkpoints experimentais

O código e as configurações de todos os experimentos NEXUS-M, Evolution Lab e AutoSynth estão versionados neste repositório.
Os checkpoints PyTorch, bancos SQLite de execuções e arquivos ZIP históricos não foram enviados: são artefatos gerados, podem exceder limites de arquivo do GitHub e devem ser distribuídos por Git LFS ou GitHub Releases.

Para reproduzir:
```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python -m nexus_m.autosynth_cli search --simulations 96 --output runs/autosynth
```

Os números dos relatórios se referem a execuções anteriores, não à execução automática do GitHub.
