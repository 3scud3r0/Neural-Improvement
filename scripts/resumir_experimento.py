"""Turn structured benchmark CSV into a Markdown report with evidence links."""
import argparse,csv,datetime,statistics
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('directory')
a=p.parse_args();directory=Path(a.directory)
path=directory/'summary.csv'
with path.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
lines=['# Relatório experimental NEXUS-M','',
       'Produzido em '+datetime.datetime.now(datetime.timezone.utc).isoformat(),'',
       'Métricas medidas neste experimento, sem extrapolação para outros problemas.','',
       '| Escala | Cenário | Acurácia OOD exata | Simulações médias | Tempo treino (s) |',
       '|---:|---|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['width']} | {r['scenario']} | {float(r['ood_exact']):.3f} | "+
                 f"{float(r['simulations']):.1f} | {float(r['train_sec']):.1f} |")
lines.extend(['','**Limitações:** poucos episódios e sementes produzem alta variância; '
             'mesmo número de passos não significa mesmo custo em FLOPs.'])
(directory/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(directory/'report.md')
