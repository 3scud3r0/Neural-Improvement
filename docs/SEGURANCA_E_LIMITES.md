# Segurança, governança e limitações

## Princípio central

Nenhum arquivo Python pode provar alinhamento de valores humanos em todas as
futuras auto-modificações de um sistema geral. O módulo `alignment_and_value_core`
implementa apenas controles operacionais testáveis em um protótipo local.

## O que está efetivamente bloqueado por padrão

- Requisição ou provisionamento de serviços em nuvem.
- Rede externa e execução nativa arbitrária.
- Alteração de código fora do diretório relativo `managed/*.py`.
- Promoção automática de uma modificação, salvo se a política for editada
  intencionalmente por um operador e as verificações passarem.
- MCTS e expansão topológica acima dos limites parametrizados.
- Código Python arbitrário no interpretador: somente AST restrita e
  orçamento finito.

## O que o sistema **não** garante

- Isolamento forte de processos sem container/hypervisor/quotas do SO.
- Auditoria imutável para quem dispõe de acesso administrativo ao disco.
- Ausência de bugs em componentes PyTorch, SQLite, SymPy ou NetworkX.
- Um algoritmo corrigido em inputs finitos funcionar para qualquer input.
- Ausência de erros de modelagem causal, de especificação de objetivos ou
  de danos indiretos em aplicações reais.
- Aprendizagem autônoma de valores incorruptíveis, nem conservação de valores
  após alterações arbitrárias futuras.

## Recomendações de uso

Executar em ambiente isolado, usar somente dados e código confiáveis, nunca
fornecer segredos, credenciais ou acesso de produção ao protótipo. Para testar
compiladores/provadores externos, usar uma máquina isolada ou container com
restrições do sistema operacional. Não carregar checkpoints de fonte não
confiável. Não ligar autonomamente ações físicas ou financeiras ao agente.

## Autoaperfeiçoamento

Um único ciclo compara dois algoritmos já revisados. O candidato não pode
alterar o guardião de segurança. Testes de correção e benchmark devem passar,
mas mesmo assim o resultado não é uma prova de equivalência universal;
os dois algoritmos específicos têm semântica simples, totalmente definida
para listas finitas de floats comuns (NaNs não fazem parte do domínio testado).
A revisão humana é necessária para promover alterações por padrão.

## Escalabilidade física

Computação, memória, I/O, energia, largura de banda e termodinâmica são limites
reais. O projeto não assume acesso ilimitado a hardware nem suprime custos
exponenciais de problemas indecidíveis ou combinatórios difíceis.
