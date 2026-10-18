# 100 redes × 100, 1.000 e 10.000 neurônios: comparação experimental

## Finalidade
Medir como o número de neurônios de cada agente altera a detecção de uma falha coordenada simulada. Há 100 agentes em cada arquitetura. O experimento usa apenas dados sintéticos, não sensores reais. O código foi treinado por 1.000 passos de otimização para cada arquitetura, com lote de 8 exemplos, e cada exemplo contém 3 instantes de comunicação.

## Arquivos
- `experimento_especializacao_v2.py`: geração de dados, definição do modelo, treinamento e gráfico.
- `comparacao_resultados.json`: métricas em dados inéditos e sob mudança de distribuição.
- `comparacao_ciclos.csv`: registros dos passos de treinamento.
- `comparacao_100_1000_10000.png`: visualização do experimento.
- `coletivo_100_com.pt`, `coletivo_1000_com.pt`, `coletivo_10000_com.pt`: pesos aprendidos.
- `avaliacao_adicional.py`, `avaliacao_calibrada.json`: calibração de limiar em conjunto independente.

## Instalação e execução
```bash
pip install numpy torch matplotlib scipy scikit-learn
python experimento_especializacao_v2.py --widths 100 1000 10000 --steps 1000 --batch 8 --test 1000 --solo-small --out resultados
python avaliacao_adicional.py --dir resultados
```
É possível começar com `--widths 100 --steps 100` para testar a instalação.

## Regras matemáticas
- `x`: `B × 3 × 100 × 3`, para lote `B`, 3 tempos, 100 sensores e 3 canais por sensor.
- `H`: 100, 1.000 ou 10.000 ativações ocultas por rede.
- `h_i(t) = tanh(x_i(t) W_i + b_i)`: ativação local de cada agente (seus pesos são diferentes).
- `s_i(t) = tanh(h_i(t) P_i / 2)`: mensagem aprendida com apenas quatro componentes; a matriz `P_i` transforma `H` em quatro números.
- `α_j = 100 softmax(a)_j`: peso de envio aprendido do agente `j`.
- `m_i(t) = Σ_(j≠i) α_j s_j(t) / Σ_(j≠i) α_j`: média ponderada de todas as mensagens, excluindo a própria.
- `u_i(t) = .65 u_i(t-1) + .35 tanh(r_i(t) + m_i(t) + c_i)`: memória temporal de quatro números, preservada entre os três tempos do mesmo exemplo.
- Produto cruzado `r_i × m_i`: ajuda a detectar covariância entre sensores.
- Camadas de leitura e uma pequena rede de decisão aprendem a classificar uma falha.
- Entropia cruzada binária: mede discrepância entre probabilidades de falha e rótulos sintéticos; AdamW ajusta os parâmetros para reduzir a discrepância.

## Controles e limites
- Teste separado: 1.000 episódios, não utilizados em treino.
- Teste de ambiente modificado: uma mudança sintética nos parâmetros de geração do ruído e da falha.
- Ablação: avaliar com mensagens desligadas, **sem retreinamento**.
- O controle de treinamento isolado foi executado apenas para a escala H=100.
- Os 1.000 ciclos são passos de otimização. Cada exemplo é processado em 3 passos temporais, portanto a comunicação não é uma rodada única em cada passo de treinamento.
- Os 10.000 neurônios por rede são ativações de uma camada larga. **Não** há uma matriz densa H × H por agente; a comunicação foi comprimida a 4 dimensões, visando viabilidade em CPU.
- A memória se mantém **dentro do episódio** e é reiniciada entre episódios independentes para evitar vazamento de informação.
- Diferentes graus de desigualdade nos pesos de transmissão não demonstram, sozinhos, papéis funcionais especializados.
- Uma única semente não permite estimar a variabilidade entre diferentes inicializações. A tarefa é criada de modo que um sensor individual não consiga inferir o rótulo; esta propriedade não representa todos os problemas reais.
