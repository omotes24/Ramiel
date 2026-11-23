# subs.py -- the path substitutions applied to the GPU-side files (code/) when they were copied from the GPU server into this
#   repository: the project root and the Python interpreter of the server are replaced by neutral ones.  Every substitution is exact,
#   and it is the only change: undoing it gives the files that were run (their md5 sums are listed in code/README.md).
SUBS = [
    ('PY=~/miniforge3/envs/dg/bin/python', 'PY=${PY:-python}'),
    ('~/dragon_glider', '~/ramiel'),
    ('$HOME/dragon_glider', '$HOME/ramiel'),
    ('{HOME}/dragon_glider', '{HOME}/ramiel'),
]

if __name__ == '__main__':
    import sys
    for p in sys.argv[1:]:
        s = open(p, encoding='utf-8').read()
        for a, b in SUBS: s = s.replace(a, b)
        open(p, 'w', encoding='utf-8', newline='').write(s)
