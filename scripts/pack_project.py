"""Package research project, consolidate readable source and verify ZIP CRCs.

Optional prior release is included in full package under legacy; source-only
archive omits its large historical checkpoint tree.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,shutil,zipfile,time

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT.parent
COMBINED=DATA/'NEXUS_EVOLUTION_CODIGO_TOTAL.txt'
FULL=DATA/'NEXUS_EVOLUTION_LAB_COMPLETO.zip'
SLIM=DATA/'NEXUS_EVOLUTION_LAB_CODIGO.zip'
NAME='NEXUS_EVOLUTION_LAB'
SOURCE_SUFFIX={'.py','.json','.toml','.md','.txt','.cfg','.yaml','.yml'}
SKIP_PARTS={'__pycache__','.pytest_cache','.git','.egg-info'}


def iter_project_files():
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or any(part in SKIP_PARTS or part.endswith('.egg-info') for part in p.relative_to(ROOT).parts):
            continue
        if p.name in ('MANIFEST_SHA256.json','NEXUS_EVOLUTION_CODIGO_TOTAL.txt'):
            continue
        yield p


def consolidate(paths):
    lines=['NEXUS EVOLUTION LAB — TODO O CÓDIGO-FONTE, CONFIGURAÇÕES E DOCUMENTAÇÃO',
           'Artefato gerado pelo empacotador; arquivos binários não são interpretados como código.',
           'Origem: módulo atual + código do lançamento anterior incluído sob legacy.',
           'Atenção: isto é código-fonte reunido em texto, não um executável compilado.','']
    count=0
    for p in paths:
        if p.suffix.lower() not in SOURCE_SUFFIX:continue
        try:content=p.read_text(encoding='utf-8')
        except (UnicodeError,PermissionError):continue
        if len(content)>750_000:continue
        relative=p.relative_to(ROOT)
        if str(relative)=='NEXUS_M_CODIGO_COMPLETO.txt':
            continue  # contents already included through individual source files
        if str(relative).endswith('/NEXUS_M_CODIGO_COMPLETO.txt'):
            continue
        lines.extend(['','='*110, f'ARQUIVO: {relative.as_posix()}',
             f'SHA256: {hashlib.sha256(content.encode()).hexdigest()}', '='*110,content,''])
        count+=1
    payload='\n'.join(lines)
    COMBINED.write_text(payload,encoding='utf-8')
    shutil.copy2(COMBINED,ROOT/COMBINED.name)
    return count,len(payload.encode())


def make_manifest(paths):
    files={p.relative_to(ROOT).as_posix():{
          'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
          for p in paths}
    txt=ROOT/COMBINED.name
    files[txt.name]={'sha256':hashlib.sha256(txt.read_bytes()).hexdigest(),
                    'bytes':txt.stat().st_size}
    info={'project':'NEXUS Evolution Lab','manifest_spec':'SHA256 relative paths; excludes this manifest',
          'files':files}
    (ROOT/'MANIFEST_SHA256.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
    return len(files)


def make_zip(destination,slim=False):
    count=0
    with zipfile.ZipFile(destination,'w',compression=zipfile.ZIP_DEFLATED,
                         compresslevel=4,allowZip64=True) as archive:
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file():continue
            relative=p.relative_to(ROOT)
            if any(part in SKIP_PARTS or part.endswith('.egg-info') for part in relative.parts):continue
            if slim and len(relative.parts)>=2 and relative.parts[:2]==('legacy','previous_nexus_m'):
                continue
            if slim and relative.parts[0]=='outputs' and p.suffix in ('.pt','.sqlite'):
                continue
            arcname=(Path(NAME)/relative).as_posix()
            archive.write(p,arcname)
            count+=1
    with zipfile.ZipFile(destination) as archive:
        corrupted=archive.testzip()
        if corrupted:raise RuntimeError(f'ZIP failed CRC: {corrupted}')
        assert len(archive.namelist())==count
    return count


def main():
    started=time.perf_counter()
    paths=list(iter_project_files())
    src_count,size=consolidate(paths)
    manifest_count=make_manifest(paths)
    full_count=make_zip(FULL,slim=False)
    slim_count=make_zip(SLIM,slim=True)
    output={'source_text_files':src_count,'combined_bytes':size,
            'manifest_files':manifest_count,
            'full_zip_files':full_count,'full_zip_bytes':FULL.stat().st_size,
            'slim_zip_files':slim_count,'slim_zip_bytes':SLIM.stat().st_size,
            'elapsed_seconds':time.perf_counter()-started,
            'artifacts':[str(FULL),str(SLIM),str(COMBINED)]}
    (DATA/'NEXUS_EVOLUTION_PACKAGING_REPORT.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
