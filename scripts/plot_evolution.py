"""Render two separate, reproducible figures from actual benchmark CSV."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path('outputs/evolution_benchmark')
    with (root/'results.csv').open(encoding='utf-8') as f:rows=list(csv.DictReader(f))
    conditions=('uniform','neural_untrained','latent_only','collective_online')
    labels=('MCTS uniforme','Redes aleatórias','Preditor latente','Coletivo online')
    percentages=[100*sum(int(row['holdout_exact']) for row in rows if row['condition']==c)/
                 max(1,sum(row['condition']==c for row in rows)) for c in conditions]
    fig,ax=plt.subplots(figsize=(9,5))
    bars=ax.bar(labels,percentages)
    for b,v in zip(bars,percentages):ax.text(b.get_x()+b.get_width()/2,v+1,f'{v:.1f}%',ha='center')
    ax.set_ylim(0,105);ax.set_ylabel('Soluções generalizadas (%)');ax.set_title('NEXUS Evolution: 24 tarefas sintéticas / 3 sementes')
    ax.tick_params(axis='x',rotation=14)
    fig.tight_layout();fig.savefig(root/'acuracia_tarefas.png',dpi=150);plt.close(fig)
    sims=[sum(int(row['mcts_simulations']) for row in rows if row['condition']==c)/
          max(1,sum(row['condition']==c for row in rows)) for c in conditions]
    fig,ax=plt.subplots(figsize=(9,5))
    bars=ax.bar(labels,sims)
    for b,v in zip(bars,sims):ax.text(b.get_x()+b.get_width()/2,v+2,f'{v:.1f}',ha='center')
    ax.set_ylim(0,max(sims)*1.22);ax.set_ylabel('Simulações MCTS médias');ax.set_title('Custo de busca (exclui treinamento e priors)')
    ax.tick_params(axis='x',rotation=14)
    fig.tight_layout();fig.savefig(root/'simulacoes_mcts.png',dpi=150);plt.close(fig)

if __name__=='__main__':main()
