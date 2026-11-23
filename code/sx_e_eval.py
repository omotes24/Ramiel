# sx_e_eval.py -- PROTOCOL_SX.md, addendum E: the one evaluation on test2 and testc (seeds 0-2) of the detector with ONE forward pass and
#   NO training (terms: code/sx_e_terms.py).   J_g = d_g * min_R{(2 lu_R - lo_R) - mean} for the two descriptors of the pass, g = f, z.
#     E2  (primary)  J_f + J_z, the pass stops at the exit block X (z = amplitude profile of the blocks 1..X)
#     E1             J_f + J_z, the pass stops at the upright block U (z of the blocks 1..U)
#     Ea             J_f + J_z, the pass stops at A, the first decoder block at twice the resolution of U
#     information    J_f only | T_f + J_z (no re-oriented f bank) | the four readings T_f + Q_f + T_z + Q_z   (z of the blocks 1..X)
#   Reference rows from stored terms: A (trained MLPs, full pass), kNN only, P1 of addendum D (density + vote, one pass to U).
#   Condition fixed by the author: one forward pass, no training of any kind, natural five-mean AUROC >= 95.3 -- read on E2, both splits.
#   Refuses to run twice.   usage: python code/sx_e_eval.py
import os, json, time
import numpy as np
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/ramiel'); OD = f'{ROOT}/runs/sx/e_final'; FOUT = f'{OD}/final.json'; assert not os.path.exists(FOUT), 'the evaluation has been run already'
TAGS = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']; SEEDS = [0, 1, 2]; t0 = time.time(); log = lambda *s: print(*s, flush=True); TARGET = 95.30
IDN = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
NAME = {'edmc10': 'CIFAR-10', 'l6svhn': 'SVHN', 'dpceleba': 'CelebA', 'in64': 'ImageNet-64', 'fmnist': 'FashionMNIST'}
def auc(x, y): r = rankdata(np.concatenate([x, y])); return 100 * float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))
def fpr95(x, y): return 100 * float((y >= np.quantile(x, 0.05)).mean())
A, K0, P1 = 'A   trained: kNN + rot MLP + mirror MLP (full pass)', 'kNN only (stored term; stops at U)', 'P1  no training, 1 pass to U: density + vote'
E2, E1, Ea = 'E2  no training, 1 pass to X: J_f + J_z', 'E1  no training, 1 pass to U: J_f + J_z', 'Ea  no training, 1 pass to A: J_f + J_z'
I1, I2, I3 = '    J_f only (1 pass to U)', '    T_f + J_z (to X; upright f bank only)', '    four readings T_f + Q_f + T_z + Q_z (to X)'
NEW = [E2, E1, Ea]; ROWS = [A, K0, P1] + NEW + [I1, I2, I3]; BR = [E2, E1]; OUT = {}
for split in ('test2', 'testc'):
    Z = np.load(f'{ROOT}/runs/ax/r/scores_{split}.npz'); SC = {}; per = {r: {t: {} for t in TAGS} for r in ROWS}
    for t in TAGS:
        pre = 'test2-' if split == 'test2' else 'testc-'; idn = pre + IDN[t]
        for sd in SEEDS:
            sets = sorted({k.split('|')[2] for k in Z.files if k.startswith(f'{t}|s{sd}|')}); Pn = np.load(f'{ROOT}/runs/sx/nt_final/p/terms_{t}_s{sd}.npz'); En = np.load(f'{OD}/terms_{t}_s{sd}.npz')
            if split == 'test2':
                H = np.load(f'{ROOT}/runs/e11/s3/heads_{t}_s{sd}.npz'); old = lambda s: tuple(H[f'{q}__{s}'].astype(np.float64) for q in ('c4', 'm'))
            else:
                T = np.load(f'{ROOT}/runs/orbc/terms_s{sd}.npz'); col = {idn + '__vflip': 6, idn + '__hflip': 4}
                old = lambda s: tuple(T[f"{t}|{idn if s in col else s}|{q}"][:, col.get(s, 0)].astype(np.float64) for q in 'CM')
            S = {}
            for s in sets:
                K = Z[f'{t}|s{sd}|{s}|K|cmin'].astype(np.float64); C_, M_ = old(s); g = lambda q: En[f'{q}|{s}'].astype(np.float64); assert len(K) == len(g('Jf')), (t, sd, s)
                S[s] = {A: K + C_ + M_, K0: K, P1: Pn[f'D|{s}'].astype(np.float64) + Pn[f'V0|{s}'].astype(np.float64), E2: g('Jf') + g('JzX'), E1: g('Jf') + g('JzU'), Ea: g('Jf') + g('JzA'),
                        I1: g('Jf'), I2: g('Tf') + g('JzX'), I3: g('Tf') + g('Qf') + g('TzX') + g('QzX')}
            SC[(t, sd)] = (idn, S)
            for r in ROWS:
                for s in sets:
                    if s != idn: per[r][t].setdefault(s, []).append((auc(S[idn][r], S[s][r]), fpr95(S[idn][r], S[s][r])))
    log(f'== {split}: natural OOD AUROC %, seeds 0-2')
    log(f'{"":54s}' + ''.join(f'{NAME[t][:11]:>12s}' for t in TAGS) + f'{"mean":>8s} |{"upside-down":>12s} |{"mirrored":>9s} |{"FPR95":>7s}'); SUM = {}
    for r in ROWS:
        nat, vf, hf, fp, pairs = [], [], [], [], {}
        for t in TAGS:
            pa = {s: float(np.mean([v[0] for v in vs])) for s, vs in per[r][t].items()}; pf = {s: float(np.mean([v[1] for v in vs])) for s, vs in per[r][t].items()}; pairs[t] = pa
            nn = [s for s in pa if '__' not in s and 'synth' not in s]; nat.append(np.mean([pa[s] for s in nn])); fp.append(np.mean([pf[s] for s in nn]))
            vf.append(np.mean([pa[s] for s in pa if s.endswith('__vflip')])); hf.append(np.mean([pa[s] for s in pa if s.endswith('__hflip')]))
        SUM[r] = dict(nat=[float(v) for v in nat], nat_mean=float(np.mean(nat)), vflip=[float(v) for v in vf], vflip_mean=float(np.mean(vf)), hflip=[float(v) for v in hf], hflip_mean=float(np.mean(hf)), fpr=[float(v) for v in fp], fpr_mean=float(np.mean(fp)), pairs=pairs)
        log(f'{r:54s}' + ''.join(f'{x:12.2f}' for x in nat) + f'{np.mean(nat):8.2f} |{np.mean(vf):12.2f} |{np.mean(hf):9.2f} |{np.mean(fp):7.1f}')
    log('   upside-down per ID dataset: ' + ' | '.join(f"{r[:4]} " + ' '.join(f'{x:.1f}' for x in SUM[r]['vflip']) for r in [A, K0, P1] + NEW))
    log('   natural five-mean per noise seed: ' + ' | '.join(f"{r[:4]} " + ' '.join(f"{np.mean([np.mean([per[r][t][s][i][0] for s in per[r][t] if '__' not in s and 'synth' not in s]) for t in TAGS]):.2f}" for i in range(len(SEEDS))) for r in [A, P1] + NEW))
    rng = np.random.RandomState(20261123); DA = {r: [] for r in BR}; AB = {r: [] for r in BR}
    for _ in range(2000):
        ta = {r: [] for r in BR}; tb = {r: [] for r in BR}
        for t in TAGS:
            da = {r: [] for r in BR}; db = {r: [] for r in BR}
            for sd in SEEDS:
                idn, S = SC[(t, sd)]; n = len(S[idn][A]); ii = rng.randint(0, n, n)
                for s in [s for s in S if s != idn and '__' not in s and 'synth' not in s]:
                    jj = rng.randint(0, len(S[s][A]), len(S[s][A])); a0 = auc(S[idn][A][ii], S[s][A][jj])
                    for r in BR: a1 = auc(S[idn][r][ii], S[s][r][jj]); da[r].append(a1 - a0); db[r].append(a1)
            for r in BR: ta[r].append(np.mean(da[r])); tb[r].append(np.mean(db[r]))
        for r in BR: DA[r].append(np.mean(ta[r])); AB[r].append(np.mean(tb[r]))
    BOOT = {}
    for r in BR:
        lo, hi = np.quantile(DA[r], [0.025, 0.975]); l2, h2 = np.quantile(AB[r], [0.025, 0.975]); BOOT[r] = dict(point=SUM[r]['nat_mean'] - SUM[A]['nat_mean'], lo=float(lo), hi=float(hi), abs_lo=float(l2), abs_hi=float(h2))
        log(f"   bootstrap, natural five-mean, [{r}]: {SUM[r]['nat_mean']:.2f} [{l2:.2f}, {h2:.2f}];  - [A]: {BOOT[r]['point']:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    OUT[split] = dict(summary=SUM, boot=BOOT); log(f'{time.time() - t0:.0f}s')
COND = {}; log(f'== condition of the author: ONE forward pass, NO training, natural five-mean AUROC >= {TARGET:.1f} (read on E2; both splits)')
for r in NEW + [P1]:
    v = {sp: OUT[sp]['summary'][r]['nat_mean'] for sp in ('test2', 'testc')}; COND[r] = dict(v, met=bool(all(x >= TARGET for x in v.values())))
    log(f"   {r:44s} test2 {v['test2']:.2f} | testc {v['testc']:.2f}   ->  {'MET' if COND[r]['met'] else 'NOT MET'}")
EQ = {}; log('== earlier rule, for information ("equal to A": natural >= A - 0.10, upside-down >= A - 1.0, FPR95 <= A + 0.5, on BOTH splits)')
for r in NEW:
    c = {sp: dict(nat=OUT[sp]['summary'][r]['nat_mean'] - OUT[sp]['summary'][A]['nat_mean'], ud=OUT[sp]['summary'][r]['vflip_mean'] - OUT[sp]['summary'][A]['vflip_mean'], fpr=OUT[sp]['summary'][r]['fpr_mean'] - OUT[sp]['summary'][A]['fpr_mean']) for sp in ('test2', 'testc')}
    ok = {k: all(f(c[sp][k]) for sp in c) for k, f in (('nat', lambda v: v >= -0.10), ('ud', lambda v: v >= -1.0), ('fpr', lambda v: v <= 0.5))}; EQ[r] = dict(diff=c, ok=ok, equal=all(ok.values()))
    log(f"   {r:44s} natural {c['test2']['nat']:+.2f} / {c['testc']['nat']:+.2f} {'ok' if ok['nat'] else 'NO'} | upside-down {c['test2']['ud']:+.2f} / {c['testc']['ud']:+.2f} {'ok' if ok['ud'] else 'NO'} | FPR95 {c['test2']['fpr']:+.1f} / {c['testc']['fpr']:+.1f} {'ok' if ok['fpr'] else 'NO'} | equal: {EQ[r]['equal']}")
META = {f'{t}_s{sd}': json.load(open(f'{OD}/meta_{t}_s{sd}.json')) for t in TAGS for sd in SEEDS}
T = {t: json.load(open(f'{ROOT}/runs/sx/bench2_{t}.json')) for t in TAGS}; tm = {t: tuple(float(T[t]['ratio'][META[f'{t}_s0'][k]]) for k in ('i_upright', 'i_A', 'i_exit')) for t in TAGS}
log('   network time / full pass (GPU, batch 128), one pass to U | A | X: ' + ' ; '.join(f'{NAME[t]} {a:.2f} | {b:.2f} | {c:.2f}' for t, (a, b, c) in tm.items()) + '   mean ' + ' | '.join(f'{np.mean([v[i] for v in tm.values()]):.2f}' for i in range(3)))
log('   bank entries (f: 8N x 2C at U; z: 8N x blocks): ' + ' ; '.join(f"{NAME[t]} N {META[f'{t}_s0']['N']}, z of {META[f'{t}_s0']['i_exit'] + 1} blocks (U {META[f'{t}_s0']['i_upright'] + 1})" for t in TAGS))
json.dump(dict(rows=ROWS, meta=META, condition=COND, equal=EQ, time=tm, target=TARGET, **OUT), open(FOUT, 'w'), indent=1); log('SX_E_EVAL_DONE', f'{time.time() - t0:.0f}s')
