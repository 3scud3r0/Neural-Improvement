# APIs dos módulos novos — NEXUS Evolution Lab

## 1. Computação geral restrita

`BudgetedPython(ExecutionBudget(...)).run(source, x)` interpreta uma função
`def main(x)` com variáveis inteiras, `while`, `if`, retornos e operações
aritméticas definidas. Não usa Python `exec`/`eval` para os programas de
entrada. Rejeita importações, chamadas, atributos, objetos arbitrários.
É computacionalmente expressivo em princípio; os limites de combustível,
tempo e bits impedem execução ilimitada.

`bounded_equivalence(a,b,lo,hi)` verifica um domínio finito.
`prove_polynomial_identity(left,right)` aceita uma gramática estrita de
polinômios e utiliza SymPy para provar a identidade sobre inteiros.
`optional_proof_assistant_check('lean'|'coq',script,allow_external=True)`
solicita uma tentativa real em provadores instalados. `optional_compiler_check`
aceita `python`,`cpp`,`rust`,`assembly` para diagnósticos e nunca executa
binários. Ferramentas externas devem rodar em ambiente isolado de verdade.

## 2. Busca hierárquica

`HierarchicalMCTS(simulations=...,max_depth=...).search(examples)` encontra
sequências de macroações de inteiros. `GoalGraph` gerencia dependências entre
subobjetivos; `Macro` define uma transição abstrata e seus limites. Busca
PUCT com expansão progressiva, poda de ações sem efeito e execução exata.
Ainda há explosão combinatória em casos ruins; não se resolve por decreto.

## 3. Topologia dinâmica

`DynamicTopology(NeuralSociety(...))` oferece `.recruit(n)`, `.widen(n)`,
`.prune(indices)`, `.suggest(gain,grad_norm)` e `.report()`. Copia os
subtensores sobrepostos para a nova arquitetura, mas mudanças no número de
agentes modificam o roteamento, e funções completas não ficam idênticas.
O otimizador antigo deve ser reconstruído após uma mudança topológica.
`cloud_capacity_request` apenas calcula um pedido sujeito a aprovação.

## 4. Causalidade e física

`CausalGraph(edges,latent=...)` testa backdoor/frontdoor e enumera conjuntos
de ajuste. `BinarySCM(graph,functions,noise_p)` permite calcular
`P(Y|do(X=x))` por enumeração exata e estimadores do tipo frontdoor/backdoor
quando as condições gráficas permitem. A enumeração aceita até 16 variáveis
binárias. `infer_linear_damped_motion(x,v,a)` estima um modelo
`aceleração = c_x*x+c_v*v+c_0` com avaliação reservada.
A linguagem do cálculo causal completo não foi implementada.

## 5. Conhecimento conceitual

`ConceptualKnowledgeGraph(path)` grava conceitos com IDs estáveis, relações
com grau de confiança, evidência, atributos e procedência. Permite registrar
referências multimodais (`uri`,`mime_type`,`sha256`) sem obter conteúdo
automaticamente. `discover_analogies` utiliza isomorfismo dirigido de pequenos
grafos; os achados começam como **hipóteses não verificadas**. Consistência
de estrutura não garante propriedades físicas equivalentes.

## 6. Autoaperfeiçoamento controlado

`ManagedSelfImprover(AlignmentGuard(workspace)).step(approve=False)` mede
`top-k` por ordenação ou heap, testa equivalência com empates e entradas
aleatórias, cria candidato em pasta gerenciada e registra resultado.
Com `approve=True` e ganho medido maior que o limiar, faz substituição
atômica do plugin permitido, recarrega função e mantém versão anterior.
Somente duas alternativas revisadas são admitidas. Limite de cinco
rodadas por processo; não edita os sete módulos centrais automaticamente.

## 7. Alinhamento e controle operacional

`AlignmentGuard(workspace,ResourcePolicy(...)).authorize(action,details,
approved=False)` impõe limites de pesquisa local e registra trilha de auditoria
encadeada por SHA256. A escrita de plugin exige caminho `managed/*.py` e
aprovação por padrão. Cloud, rede e código nativo são bloqueados por padrão.
Isso NÃO demonstra uma ontologia universal de valores humanos nem prova
que um agente mais poderoso conservará valores; SHA256 não protege contra
administradores com acesso total ao disco.

## 8. Modelo latente, metaprendizagem e orquestração

`LatentDynamics` aprende a prever a próxima representação de uma operação
inteira utilizando codificador-alvo EMA, similar apenas em espírito à JEPA.
`distill_macro_program` altera temporária ou persistentemente os pesos do
`NeuralSociety` com base em programas consistentes com exemplos públicos.
`EvolutionSystem` conecta esses módulos, e `DistributedSearchCoordinator`
executa árvores independentes em até quatro processos locais.

## Comunicação, pesos e telemetria herdados

O modelo original continua oferecendo roteamento Top-K aprendido,
`Telemetry` com logs completos e precisão FP32/FP64. O plugin de autoaperfeiçoamento
benchmarkado é um **roteador genérico de listas** e ainda não faz hot-swap
do kernel Top-K de PyTorch. Essa distinção importa na avaliação de desempenho.
