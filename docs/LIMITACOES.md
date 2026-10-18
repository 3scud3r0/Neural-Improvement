# Limitações e custos — leitura obrigatória

1. **Não é AGI**. É um protótipo de investigação, não um modelo universal.
2. **Treinamento específico não é zero**: usa 1000 tarefas sintéticas,
   exemplos públicos e os programas geradores DURANTE o treinamento.
3. **MCTS não garante solução** mesmo com limite de profundidade grande;
   a largura da árvore cresce exponencialmente.
4. **Memória pode causar sobreajuste**: coincidência nos exemplos não implica
   generalização. A seleção prefere expressões curtas, mas isto é um viés,
   não prova.
5. **A matemática de operações inteiras é exata somente dentro de cotas**.
   Não executa código arbitrário, não prova equivalência universal.
6. **FP64/Decimal não criam competência**; uma estimativa incorreta com 100
   casas decimais continua incorreta.
7. **DecimalSociety** é demonstrador lento com gradiente truncado e sem
   suporte GPU. Não é viável para um milhão de neurônios em tempo normal.
8. **Full telemetry é enorme** em larga escala; estimar bytes previamente.
9. **O roteamento top-k é diferenciável somente para links selecionados**;
   não se demonstrou especialização conceitual real.
10. **O teste OOD é limitado a uma DSL sintética**. Generalização para
    medicina, cibersegurança, indústria ou ciência real NÃO foi medida.
11. **Carga de checkpoints `.pt`**: use apenas arquivos confiáveis, de
    preferência gerados por este projeto.
12. **O aprendizado online e recuperação de conceitos** não substituem
    políticas de acesso/segurança de dados em aplicações reais.
13. **Modelos de 10000 neurônios por agente** possuem 1 milhão de ativações
    por passagem; isso não equivale a 1 milhão de unidades biológicas.
14. **MCTS compartilhado** não gera centenas de processos independentes:
    todos os agentes colaboram na mesma avaliação neural e numa só árvore.
15. **Dados de benchmark**: uma semente de treinamento não justifica
    generalizações estatísticas amplas sobre a arquitetura.
