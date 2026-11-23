# sx_g_small.py -- PROTOCOL_SX.md, addendum G: the numbers of runs/sx/g/final_s<seeds>.json that the paper uses, as one compact line.
#   usage: python code/sx_g_small.py <seeds, e.g. 0 or 012>
import os, sys, json, hashlib
j = json.load(open(os.path.expanduser(f'~/ramiel/runs/sx/g/final_s{sys.argv[1]}.json'))); o = {}
K = {'E2': 'E2  J_f + J_z, 1 pass to X (the detector)', 'E1': 'E1  J_f + J_z, 1 pass to U', 'Jf': '    J_f only', 'Jz': '    J_z only (to X)', 'TfJz': '    T_f + J_z (to X; upright f bank only)',
     'TfTz': '    T_f + T_z (to X; no ratio)', 'Tf': '    T_f only (kNN at U, upright bank)', 'four': '    four readings T_f + Q_f + T_z + Q_z (to X)'}
for t, v in j.items():
    m = v['meta'][str(v['seeds'][0])]; r = v['rows']
    o[t] = dict(seeds=v['seeds'], n_id=v['n_id'], n_ood=list(v['n_ood'].values())[0], U=m['i_upright'] + 1, X=m['i_exit'] + 1, nB=m['n_blocks'], resU=m['res_upright'], A=round(m['A'], 4), C=m['C'],
                d={k: round(sum(v['meta'][str(s)]['d'][k] for s in v['seeds']) / len(v['seeds']), 4) for k in ('f', 'zX')},
                rows={k: [round(r[n][q], 4) for q in ('nat', 'vflip', 'hflip', 'fpr95')] for k, n in K.items()}, pairs={k: {a: round(b, 4) for a, b in r[K[k]]['pairs'].items()} for k in ('E2', 'Tf')},
                ci={k.replace(' ', '_'): [round(a, 4) for a in x] for k, x in v['ci'].items()}, cfs={k[5:-1]: [round(x[q], 4) for q in ('nat', 'vflip', 'hflip')] for k, x in v['context'].items()})
s = json.dumps(o, separators=(',', ':')); print(hashlib.md5(s.encode()).hexdigest(), len(s)); print(s)
