# sx_e1b_an.py -- PROTOCOL_SX.md, addendum F: the 12 pairs of the CFS benchmark from runs/sx/e1b/<id>_s<seed>.npz (code/sx_e1b.py), three
#   noise seeds.  AvgAUROC = mean over the 12 pairs; AvgWorstAUROC = mean over the three ID datasets of the worst pair (the summaries of
#   CFS).  Bootstrap (2,000 draws; ID and OOD test images resampled; seed 0) of AvgAUROC of E2 and of its difference to the stored
#   trained detector of 1122d (runs/ax/e1).   -> runs/sx/e1b/an.json ;   usage: python code/sx_e1b_an.py
import os, json
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/ramiel'); OD = f'{ROOT}/runs/sx/e1b'; IDS = ['cifar10', 'svhn', 'celeba']; POOL = ['cifar10', 'svhn', 'celeba', 'cifar100', 'dtd']; SEEDS = (0, 1, 2); log = lambda *s: print(*s, flush=True)
def auc(x, y): r = rankdata(np.concatenate([x, y])); return float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))
Z = {(i, sd): np.load(f'{OD}/{i}_s{sd}.npz') for i in IDS for sd in SEEDS}; META = {(i, sd): json.load(open(f'{OD}/{i}_s{sd}.json')) for i in IDS for sd in SEEDS}; K = {(i, sd): np.load(f'{ROOT}/runs/ax/e1/{i}_s{sd}.npz') for i in IDS for sd in SEEDS}
new = lambda ks: (lambda i, sd, nm: sum(Z[(i, sd)][f'{k}|{nm}'] for k in ks)); old = lambda ks: (lambda i, sd, nm: sum(K[(i, sd)][f'{k}|{nm}'].astype(np.float64) for k in ks))
ROWS = {'E2': new(('Jf', 'JzX')), 'E1': new(('Jf', 'JzU')), 'Ea': new(('Jf', 'JzA')), 'Jf': new(('Jf',)), 'JzX': new(('JzX',)), 'Tf': new(('Tf',)), 'Tf+JzX': new(('Tf', 'JzX')), 'four': new(('Tf', 'Qf', 'TzX', 'QzX')),
        'A trained (stored)': old('KCM'), 'A kNN term (stored)': old('K')}
PAIRS = [(i, o) for i in IDS for o in POOL if o != i]; OUT = {'block': {i: META[(i, 0)]['label'] for i in IDS}, 'i_exit': {i: META[(i, 0)]['i_exit'] for i in IDS}, 'i_A': {i: META[(i, 0)]['i_A'] for i in IDS},
       'd': {i: [META[(i, sd)]['d'] for sd in SEEDS] for i in IDS}, 'n_sets': META[(IDS[0], 0)]['n_sets'], 'rows': {}}
for row, f in ROWS.items():
    pr = {f'{i}>{o}': float(np.mean([auc(f(i, sd, i), f(i, sd, o)) for sd in SEEDS])) for i, o in PAIRS}
    OUT['rows'][row] = dict(pairs=pr, avg=float(np.mean(list(pr.values()))), avg_worst=float(np.mean([min(pr[f'{i}>{o}'] for o in POOL if o != i) for i in IDS])), per_id={i: float(np.mean([pr[f'{i}>{o}'] for o in POOL if o != i])) for i in IDS})
rng = np.random.RandomState(20261123); n = OUT['n_sets']; B, Dd = [], []
for _ in range(2000):
    W = {s: rng.randint(0, n[s], n[s]) for s in POOL}
    e = np.mean([auc(ROWS['E2'](i, 0, i)[W[i]], ROWS['E2'](i, 0, o)[W[o]]) for i, o in PAIRS]); a_ = np.mean([auc(ROWS['A trained (stored)'](i, 0, i)[W[i]], ROWS['A trained (stored)'](i, 0, o)[W[o]]) for i, o in PAIRS]); B.append(e); Dd.append(e - a_)
OUT['boot_avg_seed0'] = [float(v) for v in np.quantile(B, [0.025, 0.975])]; OUT['boot_diff_to_trained_seed0'] = [float(v) for v in np.quantile(Dd, [0.025, 0.975])]
json.dump(OUT, open(f'{OD}/an.json', 'w'), indent=1)
log('upright block', OUT['block'], '| exit', OUT['i_exit'], '| d', {i: {k: round(v, 1) for k, v in OUT['d'][i][0].items()} for i in IDS})
log(f'{"":22s}' + ''.join(f'{i[:3]}>{o[:5]:6s}' for i, o in PAIRS) + '   Avg  AvgWorst')
for row in ROWS: r = OUT['rows'][row]; log(f'{row:22s}' + ''.join(f"{100 * r['pairs'][f'{i}>{o}']:10.2f}" for i, o in PAIRS) + f"{100 * r['avg']:7.2f}{100 * r['avg_worst']:8.2f}")
log('bootstrap 95% interval of AvgAUROC (E2, seed 0):', OUT['boot_avg_seed0'], '| E2 - trained detector:', OUT['boot_diff_to_trained_seed0']); log('SX_E1B_AN_DONE')
