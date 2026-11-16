"""Package code and executed outputs, excluding local environments and caches."""
from pathlib import Path
import hashlib,zipfile
ROOT=Path(__file__).resolve().parents[1]
def include(p):
    rel=p.relative_to(ROOT)
    return p.is_file() and not any(x in {'.venv','__pycache__','.pytest_cache','.ipynb_checkpoints','.git'} or x.endswith('.egg-info') for x in rel.parts) and p.name!='SHA256SUMS.txt'
files=sorted(p for p in ROOT.rglob('*') if include(p))
manifest=ROOT/'SHA256SUMS.txt'
manifest.write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}\n' for p in files))
archive=ROOT.parent/'Energy_Recommender_System_Complete.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files+[manifest]:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    names=z.namelist()
    for item in ['README.md','INSTALLATION.md','Research_Methodology.md','src/pipeline.py','data/synthetic/synthetic_meter.csv','results/logs/pytest.txt']:
        assert ROOT.name+'/'+item in names
print(f'{archive}\n{len(names)} files; {archive.stat().st_size:,} bytes; ZIP integrity verified')
