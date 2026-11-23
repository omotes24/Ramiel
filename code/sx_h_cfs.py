# sx_h_cfs.py -- PROTOCOL_SX.md, addendum H: runs code/cfs_extract_x.py (= cfs_extract.py with the probe batch in chunks) unchanged for the
#   tag 'adm256c' (ImageNet 256x256, ADM unconditional) on the test sets of PROTOCOL_TC3.md.  The script reads runs/ok/sets_<tag>.json,
#   which does not exist for this tag: the read is redirected to runs/tc3/sets_adm256c.json (use --prefix te-).  Batch 8 (fp32 on an
#   11-GB GPU), as for lsb256 in PROTOCOL_LSUN.md.  No frozen file is edited.
#   usage: DG_FP16=0 python code/sx_h_cfs.py --tag adm256c --gpu 0 --seed 0 --lam 7 --out runs/sx/h/cfs_l7 --prefix te-
import os, sys, runpy, builtins
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import dgcore, ct_extract_defs
spec, bank, _ = ct_extract_defs.SPEC['adm256c']; ct_extract_defs.SPEC['adm256c'] = (spec, bank, 8); os.environ.setdefault('CFS_PROBE_CHUNK', '4')
_open = builtins.open
def open_(f, *a, **k):
    if isinstance(f, str) and f == f'{dgcore.ROOT}/runs/ok/sets_adm256c.json': f = f'{dgcore.ROOT}/runs/tc3/sets_adm256c.json'
    return _open(f, *a, **k)
builtins.open = open_
sys.argv = ['cfs_extract_x.py'] + sys.argv[1:]; runpy.run_path(os.path.join(HERE, 'cfs_extract_x.py'), run_name='__main__')
