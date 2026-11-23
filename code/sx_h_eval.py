# sx_h_eval.py -- PROTOCOL_SX.md, addendum H: CFS (cfs_extract_x.py / cfs_score.py unchanged; lambda = 7, fp32; bank = the 10,000 bank
#   images) on the test images of adm256c, beside the detector of addendum E on the same images (its terms are recomputed with the
#   functions of code/sx_g_eval.py and must reproduce runs/sx/g/final_s012.json).  AUROC of te-id against every te-ood set (natural =
#   their mean), against the upside-down and the mirrored te-id images; FPR95; noise seeds 0-2 averaged; paired bootstrap (2,000 draws)
#   of CFS and of E2 - CFS.   usage: python code/sx_h_eval.py <gpu>     -> runs/sx/h/final.json  (refuses to run twice)
import os, sys, json
gpu = sys.argv[1]; src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sx_g_eval.py')).read().split('\nfor tag in TAGS:')[0]
sys.argv = ['sx_g_eval.py', gpu, '0,1,2', 'dry']; exec(compile(src, 'sx_g_eval.py', 'exec'))          # the definitions only: auc, fpr95, ldist, terms_of, E2, SEEDS, ROOT
FOUT = f'{ROOT}/runs/sx/h/final.json'; assert not os.path.exists(FOUT), 'the evaluation has been run already'; tag = 'adm256c'; CD = f'{ROOT}/runs/sx/h/cfs_l7'
REF = json.load(open(f'{ROOT}/runs/sx/g/final_s012.json'))[tag]['rows'][E2]; rows = {r[0]: np.asarray(r[2], np.int64) for r in json.load(open(f'{ROOT}/runs/tc3/sets_{tag}.json'))}
SS = {sd: terms_of(tag, sd)[0] for sd in SEEDS}; names = list(SS[SEEDS[0]]); idn = 'te-id'; ood = [n for n in names if n.startswith('te-ood-')]; vf, hf = 'te-id__vflip', 'te-id__hflip'; C1, C2 = 'CFS (diag)', 'CFS (kNN head)'
for sd in SEEDS:
    for n in names:
        z = np.load(f'{CD}/s{sd}/{tag}/{n}__scores.npz'); assert np.array_equal(z['idx'], rows[n]) and len(z['diag']) == len(SS[sd][n][E2]), (sd, n)
        SS[sd][n][C1] = z['diag'].astype(np.float64); SS[sd][n][C2] = z['knn'].astype(np.float64)                 # stored ID-high
RW = [E2, C1, C2]
A = lambda r, n, I=None, J=None, S=SEEDS: float(np.mean([auc(SS[sd][idn][r][I] if I is not None else SS[sd][idn][r], SS[sd][n][r][J] if J is not None else SS[sd][n][r]) for sd in S]))
R = {r: dict(pairs={n[7:]: A(r, n) for n in ood}, vflip=A(r, vf), hflip=A(r, hf), fpr95=float(np.mean([fpr95(SS[sd][idn][r], SS[sd][n][r]) for sd in SEEDS for n in ood])),
             per_seed=[float(np.mean([A(r, n, S=[sd]) for n in ood])) for sd in SEEDS]) for r in RW}
for r in RW: R[r]['nat'] = float(np.mean(list(R[r]['pairs'].values())))
assert abs(R[E2]['nat'] - REF['nat']) < 1e-6 and abs(R[E2]['vflip'] - REF['vflip']) < 1e-6, (R[E2]['nat'], REF['nat'])
rng = np.random.RandomState(0); nid = len(SS[SEEDS[0]][idn][E2]); B = {k: [] for k in (C1, C2, 'E2-' + C1, 'E2-' + C2)}
for _ in range(2000):
    I = rng.randint(0, nid, nid); Js = {n: rng.randint(0, len(SS[SEEDS[0]][n][E2]), len(SS[SEEDS[0]][n][E2])) for n in ood}; m = lambda r: float(np.mean([A(r, n, I, Js[n]) for n in ood])); e2 = m(E2)
    for c in (C1, C2): v = m(c); B[c].append(v); B['E2-' + c].append(e2 - v)
CI = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in B.items()}; META = {sd: json.load(open(f'{CD}/s{sd}/{tag}/meta.json')) for sd in SEEDS}
log(f'\n== ImageNet 256 (ADM, unconditional) | CFS at lambda = 7, hooks ' + ' ; '.join(f's{sd} ' + ' + '.join(META[sd]['hooks']) for sd in SEEDS) + f' | seeds {SEEDS} | ID {nid}')
log(f'{"":44s}{"natural":>8s}{"up-down":>8s}{"mirror":>8s}{"FPR95":>7s}   ' + ' '.join(f'{n[7:][:10]:>10s}' for n in ood) + '   natural per seed')
for r in RW: log(f'{r:44s}{R[r]["nat"]:8.2f}{R[r]["vflip"]:8.2f}{R[r]["hflip"]:8.2f}{R[r]["fpr95"]:7.1f}   ' + ' '.join(f'{R[r]["pairs"][n[7:]]:10.2f}' for n in ood) + '   ' + ' '.join(f'{v:.2f}' for v in R[r]['per_seed']))
log('95% intervals (paired bootstrap, 2,000 draws): ' + ' | '.join(f'{k} [{v[0]:.2f}, {v[1]:.2f}]' for k, v in CI.items()))
json.dump(dict(tag=tag, seeds=SEEDS, rows=R, ci=CI, hooks={str(sd): META[sd]['hooks'] for sd in SEEDS}, lam=META[SEEDS[0]]['lam']), open(FOUT, 'w'), indent=1); log('SX_H_EVAL_DONE', FOUT, f'{time.time() - t0:.0f}s')
