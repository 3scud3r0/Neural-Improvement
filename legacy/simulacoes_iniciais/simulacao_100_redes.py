"""Simulação reproduzível de 100 redes recorrentes, cada uma com 100 neurônios.

Em cada ciclo, toda rede envia seu vetor de 100 ativações às outras 99.
A atualização é síncrona; os pesos aleatórios ficam congelados (não há treino).
Execute: python simulacao_100_redes.py
Requisitos: numpy, matplotlib
"""
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")  # Gera o gráfico em arquivo, inclusive sem ambiente gráfico.
import matplotlib.pyplot as plt
import numpy as np

QUANTIDADE_REDES = 100
NEURONIOS_POR_REDE = 100
CICLOS = 10
SEMENTE = 2026

# Fatores de influência na atualização de cada neurônio.
FORCA_INTERNA = 0.85
FORCA_COMUNICACAO = 0.90
FORCA_ESTIMULO_COMUM = 0.12
FORCA_VIES_INDIVIDUAL = 0.12

PASTA_SAIDA = Path(__file__).resolve().parent


def concordancia(estados: np.ndarray) -> float:
    """Cosseno médio entre os vetores de ativações de pares distintos."""
    comprimentos = np.linalg.norm(estados, axis=1, keepdims=True)
    unitarios = estados / np.maximum(comprimentos, 1e-12)
    similaridades = unitarios @ unitarios.T
    n = estados.shape[0]
    return float((similaridades.sum() - np.trace(similaridades)) / (n * (n - 1)))


def diversidade(estados: np.ndarray) -> float:
    """Distância RMS média, por neurônio, até o estado médio da população."""
    diferenca = estados - estados.mean(axis=0, keepdims=True)
    return float(np.sqrt(np.mean(diferenca**2)))


def simular(estados_iniciais, pesos, vieses, estimulo, comunicacao=True):
    """Devolve estados e métricas do instante inicial até o ciclo 10."""
    estados = estados_iniciais.copy()
    historico = [estados.copy()]
    metricas = []

    for ciclo in range(CICLOS + 1):
        metricas.append({
            "ciclo": ciclo,
            "concordancia": concordancia(estados),
            "diversidade": diversidade(estados),
            "amplitude_rms": float(np.sqrt(np.mean(estados**2))),
        })
        if ciclo == CICLOS:
            break

        # Produto interno: para cada rede, matriz 100x100 vezes vetor 100.
        processamento_interno = np.einsum("nij,nj->ni", pesos, estados)

        if comunicacao:
            # Cada rede recebe a média das ativações das OUTRAS 99 redes.
            # A fórmula compacta é exatamente equivalente a somar 99 mensagens.
            mensagens_das_outras = (
                estados.sum(axis=0, keepdims=True) - estados
            ) / (QUANTIDADE_REDES - 1)
        else:
            mensagens_das_outras = np.zeros_like(estados)

        # Atualização simultânea: nenhuma rede vê mensagens do ciclo seguinte.
        estados = np.tanh(
            FORCA_INTERNA * processamento_interno
            + FORCA_COMUNICACAO * mensagens_das_outras
            + FORCA_ESTIMULO_COMUM * estimulo
            + FORCA_VIES_INDIVIDUAL * vieses
        )
        historico.append(estados.copy())

    return np.asarray(historico), metricas


def main():
    rng = np.random.default_rng(SEMENTE)
    n, m = QUANTIDADE_REDES, NEURONIOS_POR_REDE

    # Cada rede tem pesos recorrentes próprios, fixos e aleatórios.
    pesos = rng.normal(0, 1 / np.sqrt(m), size=(n, m, m))
    estados_iniciais = rng.normal(0, 0.50, size=(n, m))
    vieses = rng.normal(0, 1, size=(n, m))
    estimulo = rng.normal(0, 1, size=(1, m))  # Mesmo estímulo para todas.

    hist_com, dados_com = simular(estados_iniciais, pesos, vieses, estimulo, True)
    hist_sem, dados_sem = simular(estados_iniciais, pesos, vieses, estimulo, False)

    arquivo_csv = PASTA_SAIDA / "metricas_100_redes.csv"
    with arquivo_csv.open("w", newline="", encoding="utf-8") as arquivo:
        escritor = csv.writer(arquivo)
        escritor.writerow([
            "ciclo", "concordancia_com", "concordancia_sem",
            "diversidade_com", "diversidade_sem", "rms_com", "rms_sem",
        ])
        for a, b in zip(dados_com, dados_sem):
            escritor.writerow([
                a["ciclo"], a["concordancia"], b["concordancia"],
                a["diversidade"], b["diversidade"],
                a["amplitude_rms"], b["amplitude_rms"],
            ])

    # Gráficos de métricas e de ativações finais de 20 redes.
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ciclos = [r["ciclo"] for r in dados_com]
    for nome, registros in [("Com comunicação", dados_com), ("Sem comunicação", dados_sem)]:
        axs[0, 0].plot(ciclos, [r["concordancia"] for r in registros], marker="o", label=nome)
        axs[0, 1].plot(ciclos, [r["diversidade"] for r in registros], marker="o", label=nome)
    axs[0, 0].set(title="Semelhança média entre pares", xlabel="Ciclo", ylabel="Cosseno médio", ylim=(-0.1, 1.05))
    axs[0, 1].set(title="Diversidade de estados", xlabel="Ciclo", ylabel="Desvio RMS por neurônio")
    for ax in axs[0]:
        ax.grid(alpha=0.25)
        ax.legend()

    for ax, matriz, titulo in [
        (axs[1, 0], hist_com[-1, :20], "Com comunicação — ciclo 10"),
        (axs[1, 1], hist_sem[-1, :20], "Sem comunicação — ciclo 10"),
    ]:
        im = ax.imshow(matriz, cmap="coolwarm", vmin=-0.8, vmax=0.8, aspect="auto")
        ax.set(title=titulo, xlabel="Neurônio (0–99)", ylabel="Rede (0–19)")
        fig.colorbar(im, ax=ax, shrink=0.75, label="Ativação")
    arquivo_png = PASTA_SAIDA / "resultado_100_redes.png"
    fig.savefig(arquivo_png, dpi=160)
    plt.close(fig)

    print("REDES:", n, "NEURÔNIOS/REDE:", m, "CICLOS:", CICLOS)
    print("Mensagens dirigidas por ciclo:", n * (n - 1))
    print("Mensagens dirigidas em 10 ciclos:", CICLOS * n * (n - 1))
    print("Valores neuronais transmitidos em 10 ciclos:", CICLOS * n * (n - 1) * m)
    print("\nCiclo | Concordância (com / sem) | Diversidade (com / sem)")
    for i in range(CICLOS + 1):
        a, b = dados_com[i], dados_sem[i]
        print(f"{i:>5} | {a['concordancia']:>8.4f} / {b['concordancia']:<8.4f} | {a['diversidade']:>8.4f} / {b['diversidade']:<8.4f}")
    print("\nArquivos:", arquivo_png, arquivo_csv, sep="\n")


if __name__ == "__main__":
    main()
