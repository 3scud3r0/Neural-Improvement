# Obrigações formais ainda não resolvidas

Não considerar testes finitos um teorema sobre todos os programas. Para prover garantias mais fortes, são necessárias formalizações em Lean/Coq, especificações do Python e hipóteses explícitas.

1. Especificação matemática de top-k: para todos os scores finitos e `0 <= k <= n`, o resultado contém `k` índices distintos, ordenados por `(-scores[i], i)` e iguais aos primeiros `k` da ordem total.
2. Provar que cada algoritmo selecionado (`sorted`, `heapq.nsmallest`, `scan`) implementa essa relação sob as hipóteses da linguagem e da biblioteca padrão.
3. Provar terminação e limites de memória sob entradas finitas; tratar explicitamente números não finitos, objetos Python arbitrários e `k` inválido.
4. Relacionar a semântica do código AST à semântica de um modelo formal suficientemente preciso para tornar relevante o certificado.
5. Verificar protocolos de promoção, backup e recuperação sob falhas de disco/processo.

Nenhum desses resultados gerais foi produzido automaticamente neste projeto. O campo `machine_checked_universal_proof` permanece `false`, e o gate só fornece verificações de escopo limitado e argumentos estruturais.
