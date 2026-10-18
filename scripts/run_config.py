"""Launch documented JSON configurations without shell interpolation.
Usage: python scripts/run_config.py configs/benchmark_tres_escalas.json
"""
import json,subprocess,sys
from pathlib import Path


def main():
    if len(sys.argv)!=2:
        raise SystemExit('Usage: python scripts/run_config.py config.json')
    config=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    command=config['command']
    if command not in ('train','solve','benchmark','decimal-demo','demo','inspect','adapt','world-train'):
        raise SystemExit('Unsupported command')
    tokens=[sys.executable,'-m','nexus_m',command]
    for key,val in config.get('arguments',{}).items():
        option='--'+key.replace('_','-')
        if isinstance(val,bool):
            if val:tokens.append(option)
        elif isinstance(val,list):
            tokens.append(option)
            tokens.extend(str(x) for x in val)
        else:tokens.extend((option,str(val)))
    print('Running:',tokens,flush=True)
    subprocess.run(tokens,check=True)

if __name__=='__main__':main()
