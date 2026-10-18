"""Plot measured results of the AST-MCTS self-optimization experiment."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'/'autosynth'
OUT.mkdir(exist_ok=True)
trial=json.loads((ROOT/'outputs/autosynth_trial/managed/ast_search_trace.json').read_text())
first=json.loads((ROOT/'outputs/autosynth_approved_example_report.json').read_text())
second=json.loads((ROOT/'outputs/autosynth_second_iteration_report.json').read_text())
items=[x for x in trial['candidates'] if x['equivalent'] and x['baseline_ms'] and x['candidate_ms']]
items.sort(key=lambda x:x['improvement'],reverse=True)
labels=[x['spec']['strategy']+' / '+x['spec']['key'] for x in items[:8]]
values=[100*x['improvement'] for x in items[:8]]
fig,ax=plt.subplots(1,3,figsize=(15,4.3),layout='constrained')
ax[0].barh(labels[::-1],values[::-1])
ax[0].set_xlabel('Reduction in elapsed time (%)')
ax[0].set_title('Valid synthesized candidates')
ax[0].axvline(0,linewidth=.8)
trace=trial['trajectory']
ax[1].plot([t['simulation'] for t in trace],
           [100*(t['best_gain'] or 0) for t in trace])
ax[1].set_xlabel('MCTS simulations')
ax[1].set_ylabel('Best observed speed gain (%)')
ax[1].set_title('AST-search improvement curve')
ax[1].grid(alpha=.2)
ax[2].bar(['Initial optimization','Next iteration'],
          [100*first['heldout_gain'],100*second['heldout_gain']])
ax[2].axhline(5,linestyle='--',linewidth=1,label='Minimum required gain (5%)')
ax[2].set_ylabel('Held-out time reduction (%)')
ax[2].set_title('Iterative source optimization')
ax[2].legend(fontsize=8)
ax[2].tick_params(axis='x',rotation=13)
fig.savefig(OUT/'autosynth_resultados.png',dpi=170,bbox_inches='tight')
plt.close(fig)
print(OUT/'autosynth_resultados.png')
