# O que é verificado e o que não é

1. **AST segura de Python restrito:** validação sintática e execução interpretada
   com limite de tempo/instruções/inteiros. Não é prova de correção.
2. **Equivalência finita:** executa dois programas em todos os inteiros de
   `[lo,hi]`. Status `bounded_pass` significa apenas esse intervalo.
3. **Identidade polinomial:** expressões de `x`, constantes, `+`, `-`, `*`
   e potências inteiras pequenas são transformadas por um parser AST controlado
   e verificadas com álgebra simbólica exata. Status `proved` aplica-se apenas
   à identidade especificada, não a programas gerais.
4. **Lean/Coq:** adaptadores opcionalmente invocam provadores externos e
   preservam seu resultado real; não geram automaticamente provas para um
   algoritmo arbitrário. Sem o executável instalado, `proved=False`.
5. **C++/Rust/Assembly:** checagem sintática/compilação mediante opt-in.
   Compilar com sucesso não prova correção, memória segura ou ausência de
   comportamento indefinido.
6. **MCTS/verificação de programas:** uma solução que coincide com cinco
   exemplos não está matematicamente provada. Testes separados podem rejeitá-la.
7. **Alinhamento:** não há prova de valores incorruptíveis para sistemas que
   modificam arbitrariamente a própria implementação.

Em geral, correção universal exige especificações e teorias formais adequadas,
e o problema da parada exclui um decisor geral para programas arbitrários.
