"""Rebuild the consolidated TXT and ZIP from this project checkout.
Python scripts/configs only go into TXT; all non-cache files go into ZIP.
"""
from pathlib import Path
import hashlib,json,zipfile

root=Path(__file__).resolve().parents[1]
outdir=root.parent
code_path=outdir/'NEXUS_M_CODIGO_COMPLETO.txt'
zip_path=outdir/'NEXUS_M_TODOS_ARQUIVOS.zip'
EXCLUDE={'__pycache__','.pytest_cache','.mypy_cache','.ruff_cache','.git'}
files=sorted(p for p in root.rglob('*') if p.is_file() and not (set(p.relative_to(root).parts)&EXCLUDE) and p.suffix not in ('.pyc',))
source_extensions={'.py','.toml','.json','.sh','.yml','.yaml','.md'}
with code_path.open('w',encoding='utf-8',newline='\n') as out:
    out.write('NEXUS-M — TODO O CÓDIGO-FONTE CONSOLIDADO\n')
    out.write('Cada seção corresponde a um arquivo distinto na estrutura do ZIP.\n')
    out.write('Texto de leitura/consulta, NÃO é um script Python único.\n\n')
    for p in files:
        if p.suffix not in source_extensions:continue
        # Exclude output reports from source listing but keep saved model summaries in ZIP.
        if p.relative_to(root).parts[0]=='outputs':continue
        relative=p.relative_to(root).as_posix()
        try:contents=p.read_text(encoding='utf-8')
        except UnicodeDecodeError:continue
        out.write('\n'+'='*95+'\n')
        out.write(f'ARQUIVO: {relative}\n')
        out.write('='*95+'\n\n')
        out.write(contents.rstrip()+'\n')
manifest=[{'file':p.relative_to(root).as_posix(),
           'size':p.stat().st_size,
           'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in files]
manifest_path=root/'MANIFEST_SHA256.json'
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
# One stable zip; includes large older and newer checkpoint weights.
files=sorted(p for p in root.rglob('*') if p.is_file() and not (set(p.relative_to(root).parts)&EXCLUDE) and p.suffix!='.pyc')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=5,allowZip64=True) as z:
    for p in files:
        z.write(p,arcname=(root.name+'/'+p.relative_to(root).as_posix()))
    z.write(code_path,arcname=root.name+'/'+code_path.name)
with zipfile.ZipFile(zip_path) as z:
    bad=z.testzip()
    if bad:raise RuntimeError(f'Corrupt archive: {bad}')
print(json.dumps({'zip':str(zip_path),'zip_bytes':zip_path.stat().st_size,
                  'txt':str(code_path),'txt_bytes':code_path.stat().st_size,
                  'files_inside_zip':len(files)+1},indent=2))
