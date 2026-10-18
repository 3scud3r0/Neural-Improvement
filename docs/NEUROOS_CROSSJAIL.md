# NeuroOS, execução contínua e CrossJail

Esta versão integra o arquivo Python do NeuroOS e muda os padrões de promoção
do NEXUS. `ResourcePolicy()` permite capacidades, promoção automática e quotas
sem teto (`None`). A trilha de auditoria continua disponível. Para recuperar
os limites do laboratório anterior, use `ResourcePolicy.restricted()` na API.

## Instalação

Use um ambiente virtual: a jaula desabilita o diretório de pacotes do usuário,
`PYTHONPATH` e variáveis de inicialização herdadas.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,neuroos]"
python -m pytest -q
```

## Mudanças de comportamento

| Controle anterior | Comportamento atual |
|---|---|
| Cinco tentativas por processo | Sem teto global; `--max-cycles N` é opcional |
| `--approve` obrigatório | Promove automaticamente se correção e ganho passarem; `--no-approve` apenas prepara |
| Fonte ativa em `managed/*.py` | `--target` escolhe um caminho relativo ao workspace `--output` |
| Loader limitado a dois templates | `ManagedSelfImprover` carrega plugins Python que definem `route(scores, k)` |
| Quotas e capacidades negadas por padrão | Configuráveis; sem teto e permitidas por padrão |
| Máximo de 500 simulações AST | Qualquer quantidade positiva, sujeita à política configurada |

```bash
python -m nexus_m.evolution_cli improve --output runs/improve --rounds 12
python -m nexus_m.evolution_cli improve --output runs/improve --rounds 0
python -m nexus_m.evolution_cli improve --output runs/improve --rounds 0 --max-cycles 20
python -m nexus_m.evolution_cli improve --output runs/improve --target plugins/router.py
python -m nexus_m.autosynth_cli search --output runs/autosynth --target plugins/ast_router.py
python -m nexus_m.autosynth_cli search --output runs/autosynth --no-approve
python -m nexus_m.autosynth_cli rollback --output runs/autosynth --target plugins/ast_router.py
```

`--rounds 0` transmite um relatório JSON por linha até interrupção ou limite
configurado. O iterador da API é `iter_optimize(rounds=0)`; `optimize(N)` retorna
uma lista finita. `--output` define a raiz dos caminhos editáveis: para trabalhar
no próprio projeto, essa raiz deve ser o projeto. Caminhos que escapam da raiz,
inclusive por symlinks, são recusados.

A autorização de edição admite qualquer arquivo dentro dessa raiz. Os dois
otimizadores existentes continuam especializados em roteadores top-k: não
reescrevem módulos arbitrários. AutoSynth mantém sua gramática de 24 programas,
testes diferenciais, evidência finita e verificação estrutural. O improver
tradicional continua propondo sort/heap. Remover permissões não cria um gerador
universal de programas. A promoção exige o ganho medido configurado, preserva
backup e troca a fonte por substituição atômica.

## NeuroOS incorporado

Origem: [3scud3r0/NeuroOS](https://github.com/3scud3r0/NeuroOS), commit
`2a3b46b2cdb1de377b013458173ae0b3d64b3590`. O arquivo upstream é um documento
Markdown com extensão `.py`. Extraímos seu primeiro bloco Python, sem mudar o
código desse bloco. O módulo fica em `nexus_m/_vendor/neuroos.py`; licença MIT,
hashes e transformação estão ao lado dele.

A integração usa importação tardia. Importar NEXUS não instancia NeuroOS,
não baixa modelos e não inicia seus serviços. Um modelo GGUF local é necessário
para inferência. A implementação usa PyTorch/NumPy e pode usar dependências
nativas; ser um único arquivo Python não a torna independente dessas bibliotecas.

```python
from nexus_m.evolution_system import EvolutionSystem

with EvolutionSystem("runs/session") as nexus:
    text = nexus.generate_text("Explique MCTS", "models/model.gguf",
                               max_new_tokens=128)
    print(text)
```

```python
from nexus_m.neuroos_bridge import NeuroOSBridge

with NeuroOSBridge("models/model.gguf", "runs/neuroos",
                   prefer_cuda=False, dtype_name="float32") as engine:
    print(engine.generate("Explique MCTS", max_new_tokens=128))
```

O adaptador guarda Forge/Cortex no workspace e desliga a API HTTP por padrão.
Instanciar o motor inicia os workers que o upstream implementa; `close()` e o
context manager chamam `shutdown()`. A CLI original está disponível em:

```bash
python -m nexus_m.neuroos_bridge --help
python -m nexus_m.neuroos_bridge --model models/model.gguf --cpu --dtype float32 --prompt "Olá"
```

O extra `neuroos-api` instala FastAPI/Uvicorn para a API opcional. Recursos de
treinamento, mineração e embeddings têm dependências opcionais adicionais do
upstream. Importação e conexão com o orquestrador foram verificadas; inferência
GGUF, treinamento LoRA e qualidade das respostas precisam de validação com
modelo, dados e hardware reais. Não se pressupõe que todos os recursos descritos
pelo upstream sejam completos.

## CrossJail

`crossjail.py` é independente do NEXUS e usa somente a biblioteca padrão. Pode
ser copiado para outro projeto. Após instalação, também existe o comando
`crossjail`.

```bash
# A raiz deve ficar fora da árvore de origem, para evitar cópia recursiva.
python crossjail.py init --root ../jail_nexus --src . --timeout 900 --mem-mb 8192
python crossjail.py status --root ../jail_nexus
python crossjail.py run --root ../jail_nexus -- python -m nexus_m.evolution_cli improve --output runs/improve --rounds 12
python crossjail.py destroy --root ../jail_nexus --force
```

Linux/macOS também aceitam `--cmd "python -m ..."`. No Windows, use a forma
`-- python ...`, que preserva argumentos contendo espaços sem reparsá-los.

| Proteção | Implementação e limites |
|---|---|
| Cópia independente | Ignora Git, caches, ambientes virtuais e nomes comuns de credenciais; recusa symlinks; não oferece hardlinks |
| Ambiente | Allowlist pequena, HOME/TEMP próprios; remove credenciais e caminhos Python herdados; nomes incomuns de segredos em arquivos da origem exigem revisão da origem |
| Prazo | Timeout do processo; encerra grupo POSIX ou árvore Windows via `taskkill` |
| Memória/CPU | `resource`: CPU no POSIX, memória virtual no Linux; macOS não oferece o limite de memória desta implementação. Falha ao iniciar se um limite suportado solicitado não puder ser aplicado; limites são por processo, não uma quota agregada da árvore |
| Windows | Sem limite de memória/CPU por Job Object nesta versão; `status` declara essa ausência |
| Rede Linux | Prova de disponibilidade seguida de `unshare --user --map-root-user --net` para o filho |
| Rede macOS | `sandbox-exec`, quando disponível e aceito pelo sistema |
| Rede Windows | Bloqueio indisponível; execução sem permissão de rede é recusada |
| Arquivos do host | **Não há sandbox de filesystem do sistema operacional**; o payload conserva as permissões do usuário |
| Auditoria | JSONL e metadados persistentes; não são resistentes a alterações por código hostil com as mesmas permissões |
| Saída | Retorno limitado a 1 MiB por stream; spools temporários podem consumir disco |

Se o host não permitir isolamento de rede, o payload não é iniciado e o runner
retorna 125. Não há fallback silencioso. Para permitir rede explicitamente:

```bash
python crossjail.py init --root ../jail_network --src . --network
python crossjail.py exec --src . --network --timeout 300 -- python -m pytest -q
```

`--network` permite acesso à rede; não instala regras de firewall. No ambiente
cloud usado nesta implementação, o kernel recusou `unshare`; o teste confirmou
a recusa de execução, sem alegar isolamento estabelecido. Para código hostil,
use VM descartável ou sandbox real de kernel. A cópia e o ambiente limpo não
impedem leitura/escrita absoluta no host, execução nativa ou processos que
deliberadamente escapem de grupos POSIX.

`restricted_exec()` é apenas um helper para snippets pequenos: captura stdout,
aplica timeout em processo separado e usa builtins mínimos sem imports. Sua
filtragem AST não é uma fronteira de segurança Python.

`status` não cria raízes ausentes. `destroy --force` e `init --force` recusam
diretórios sem metadados válidos ou sobrepostos à origem. Execuções temporárias
apagam a cópia e o audit log ao terminar; `RunResult.audit_path` fica `None`.
O CLI retorna 124 em timeout e preserva o status de saída normal do payload.

## Verificação

```bash
python -m pytest -q
python -m compileall -q nexus_m crossjail.py
python -m nexus_m.neuroos_bridge --help
```

A suíte cobre promoção e rollback, execução além de cinco ciclos, plugin externo
a `managed/`, quotas opt-in, lifecycle NeuroOS com um motor substituto, cópia
independente, scrubbing, política persistida, rede aplicada ou execução recusada,
limite de memória Linux, timeout e descendentes, argumentos com espaços e
proteção contra exclusão/cópia recursiva. A matriz de CI executa os testes do
CrossJail em Linux, Windows e macOS; resultados dessas plataformas só existem
após o workflow rodar.
