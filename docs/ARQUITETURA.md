# Arquitetura NEXUS-M v0.1

## 1. Agentes neurais

Com N agentes de largura H, o estado oculto tem forma `[N,H]` e é REALMENTE
calculado a cada chamada. O vetor de entrada possui 48 características
normalizadas dos exemplos e do prefixo do programa. Cada rede possui vetores
próprios de ganho, viés, recorrência e viés de roteamento. O codificador,
emissor e receptor de mensagens usam projeções compartilhadas: O(NH+Hd),
não O(NH²). Não alegamos independência total de todos os pesos entre agentes.

## 2. Comunicação seletiva

Cada agente gera mensagem `m_i=tanh(W_msg h_i+b_i)` com dimensão `d`.
Produzimos consulta `q_i` e chave `k_j` com projeções aprendidas.
`score_ij=<normalize(q_i),normalize(k_j)>/sqrt(d)`, com `score_ii=-inf`.
Selecionam-se K destinos **remetentes** por receptor por `topk`, seguidos de
softmax. Cada agente integra apenas K mensagens. Com N=100 e K=4, são
N*K=400 ligações receptor-emissor ativas por rodada, contra 9.900 possíveis.

Isso implementa um roteador neural. Não garante automaticamente que agentes
se especializem funcionalmente. Existem testes interventivos em
`nexus_m/diagnostics.py`.

## 3. Estados e memória operacional

A atualização recursiva usa `tanh` com `layer_norm`, ganho e recorrência
específicos por agente. `recurrence` define as rodadas internas por avaliação.
O estado **intraavaliação** persiste de uma rodada para outra, mas normalmente
é reiniciado na avaliação de outro nó MCTS para preservar independência entre
ramificações. O MCTS mantém informações de busca entre os nós.

## 4. MCTS compartilhado

Cada nó corresponde a um prefixo de programa pós-fixado. A seleção usa:
`Q(s,a) + c*P(s,a)*sqrt(N(s))/(1+N(s,a))`.

Ao visitar um nó novo, o sistema calcula uma política prior a partir da média
ponderada das propostas das N redes, restringe tokens gramaticalmente válidos,
executa rollouts e propaga o valor terminal. Resultados parciais são
armazenados como melhor candidato. O número máximo de simulações e a
profundidade são configuráveis. Todas as redes orientam **a mesma árvore**,
não 100 árvores independentes.

## 5. Verificador e modelo exato de transição

A DSL de inteiros é uma máquina de pilha determinística e limitada. `execute`
implementa as transições exatamente sob as cotas. `HALT` define o programa
terminado. Erros e estouro retornam falha. Não há execução arbitrária de
comandos do usuário. O executor exato atua como simulador/modelo do ambiente;
há também o preditor opcional `world_model.py` que aprende transições no espaço de características dessa linguagem formal. Não existe um preditor físico universal treinado. Essa limitação importa em
ambientes onde as regras não são conhecidas.

## 6. Memória episódica, semântica e procedimental

`memory.sqlite` conserva exemplos vistos, programas propostos, resultados de
verificação NOS EXEMPLOS, procedência e timestamps. A tabela de conceitos
agrega programas aprovados nos exemplos. A recuperação ordena candidatos
pela consistência com as entradas atuais e pelo comprimento do programa.
A busca sempre reexecuta o candidato antes de aceitá-lo. Nenhum resultado
oculto entra na recuperação. Isso reduz redescoberta, mas programas que
coincidem nos poucos exemplos podem ainda ser incorretos fora deles.

## 7. Três escalas de aprendizagem

- Rápida: contexto, estados e expansão da árvore.
- Intermediária: adaptação provisória de ganhos e roteamento, sem alterar
  permanentemente os pesos padrão. É regularização não supervisionada, não um
  meta-aprendiz perfeito.
- Lenta: AdamW com exemplos sintéticos de famílias de treino, distilação
  opcional de MCTS, consolidação de experiências verificadas nos exemplos.

Os oráculos dos programas sintéticos entram SOMENTE no treinamento
supervisionado e na avaliação de teste **após** a resolução, nunca no MCTS.

## 8. Observabilidade

Eventos JSONL são gravados incrementalmente: passo, tempo UTC, normas de
estados/mensagens, entropia de influência, quantidade de canais ativos,
parâmetros e gradientes. Modo `full` salva também tensores de estados,
mensagens, roteamento e pesos com etapas indexadas.

## 9. Precisão

PyTorch disponibiliza execução do modelo em FP32 e FP64. O modo adaptativo
reserva inteiros exatos e acumuladores de 64 bits para o verificador e MCTS.
O microbackend `DecimalSociety` usa dígitos arbitrários em pesos, ativações,
mensagens e atualizações SGD (derivada truncada de uma rodada). Sua finalidade
é investigar efeito de erro de arredondamento em escala pequena.

## 10. O que não está implementado

Inteligência geral, descoberta garantida de novas leis científicas,
modificação segura e autônoma da própria arquitetura, conhecimento sem dados,
busca exaustiva em profundidade arbitrária, modelo do mundo físico completo,
linguagem emergente comprovada e verificações matemáticas gerais de programas
são objetivos de pesquisa, NÃO capacidades demonstradas neste software.
