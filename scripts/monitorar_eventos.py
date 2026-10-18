"""Tail live telemetry JSONL; run in a second terminal during training."""
import json,sys,time
from pathlib import Path
if len(sys.argv)<2:
    raise SystemExit('Usage: python scripts/monitorar_eventos.py runs/.../telemetry/events.jsonl')
p=Path(sys.argv[1]);limit=int(sys.argv[2]) if len(sys.argv)>2 else 0
seen=0
print('Monitoring:',p,flush=True)
while not p.exists():time.sleep(.5)
with p.open(encoding='utf-8') as stream:
    while True:
        line=stream.readline()
        if not line:
            if limit and seen>=limit:break
            time.sleep(.5)
            continue
        try:e=json.loads(line)
        except json.JSONDecodeError:continue
        if e.get('type') in ('training','search_complete','communication','parameters','mcts'):
            brief={k:e.get(k) for k in ('type','step','loss','simulations','best_score','messages','state_rms') if k in e}
            print(json.dumps(brief,ensure_ascii=False),flush=True)
        seen+=1
        if limit and seen>=limit:break
