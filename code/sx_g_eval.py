# sx_g_eval.py -- PROTOCOL_SX.md, addendum G: the detector of addendum E (ONE truncated forward pass, NO training) on the 256x256 ADM
#   tiers lsb256 (LSUN bedroom) and adm256c (ImageNet).  Features: code/sx_g_extract.py.  ldist() and terms() are those of
#   code/sx_e_terms.py (bank = 10,000 training images in eight orientations; mean over 5,000 of them with their own copies left out).
#     E2 (the detector)  J_f + J_z, z = amplitude profile of the blocks 1..X          E1: z of 1..U          Ea: z of 1..A
#     parts              J_f | J_z | T_f + J_z | T_f + T_z | T_f | the four readings T_f + Q_f + T_z + Q_z   (z of 1..X)
#   AUROC of the test ID images against every test OOD set (natural = their mean), against the upside-down and the mirrored test ID
#   images; FPR at 95% TPR (mean over the OOD sets); paired bootstrap, 2,000 draws.  With several seeds: scores per seed, AUROC averaged.
#   usage: python code/sx_g_eval.py <gpu> <seeds, e.g. 0 or 0,1,2>      -> runs/sx/g/final_s<seeds>.json  (refuses to run twice)
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import rankdata
ROOT = os.path.expanduser('~/ramiel'); dev = torch.device(f'cuda:{int(sys.argv[1])}'); SEEDS = [int(s) for s in sys.argv[2].split(',')]; DRY = len(sys.argv) > 3 and sys.argv[3] == 'dry'
OD = f'{ROOT}/runs/sx/g'; FOUT = f'{OD}/final_s{"".join(map(str, SEEDS))}.json'; assert DRY or not os.path.exists(FOUT), 'the evaluation has been run already'
log = lambda *s: print(*s, flush=True); t0 = time.time(); TAGS = ['lsb256', 'adm256c']; NAME = {'lsb256': 'LSUN bedroom 256 (ADM)', 'adm256c': 'ImageNet 256 (ADM, unconditional)'}
RS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]; RO = [7 * r for r in RS]; lR = np.log(RS)
t_ = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev).float(); un = lambda a: F.normalize(t_(a), dim=-1)
def auc(x, y): r = rankdata(np.concatenate([x, y])); return 100 * float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))
def fpr95(x, y): return 100 * float((y >= np.quantile(x, 0.05)).mean())
def ldist(Q, Bk, ranks, own=None, per=1, cos=False, bs=256):     # -log distance to the ranks-th nearest row of Bk; own: index of the query's own image (left out in all its `per` copies; Bk stacked copy-major)
    out = []; n2 = None if cos else (Bk * Bk).sum(1); nimg = len(Bk) // per
    for i in range(0, len(Q), bs):
        q = Q[i:i + bs]; d2 = 2 - 2 * q @ Bk.T if cos else (q * q).sum(1, keepdim=True) + n2[None] - 2 * q @ Bk.T
        if own is not None: d2.view(len(q), per, nimg)[torch.arange(len(q), device=dev), :, own[i:i + bs]] = float('inf')
        out.append(-0.5 * torch.log((-(-d2).topk(max(ranks), dim=1).values[:, [r - 1 for r in ranks]]).clamp_min(1e-12)))
    return torch.cat(out).double().cpu().numpy()
E2, E1, Ea = 'E2  J_f + J_z, 1 pass to X (the detector)', 'E1  J_f + J_z, 1 pass to U', 'Ea  J_f + J_z, 1 pass to A'
P = ['    J_f only', '    J_z only (to X)', '    T_f + J_z (to X; upright f bank only)', '    T_f + T_z (to X; no ratio)', '    T_f only (kNN at U, upright bank)', '    four readings T_f + Q_f + T_z + Q_z (to X)']
ROWS = [E2, E1, Ea] + P; RES = {}
def terms_of(tag, seed):
    FD = f'{OD}/feat/s{seed}/{tag}'; M = json.load(open(f'{FD}/meta.json')); iu, ia, ix = M['i_upright'], M['i_A'], M['i_exit']; names = M['sets']
    parts = sorted(f for f in os.listdir(FD) if f.startswith('bank_f_p')); n0 = len(parts); assert n0 == int(parts[0].split('of')[1].split('.')[0])
    UNI = np.concatenate([np.load(f'{FD}/bank_f_p{i}of{n0}.npy') for i in range(n0)]).astype(np.float32); AP = np.concatenate([np.load(f'{FD}/bank_z_p{i}of{n0}.npy') for i in range(n0)])
    chk = max(json.load(open(f'{FD}/meta_bank_p{i}of{n0}.json'))['check_identity_vs_tc3_bank'] for i in range(n0)); N, _, D_ = UNI.shape; assert N == 10000 and AP.shape == (N, 8, ix + 1)
    assert np.isfinite(UNI).all() and np.isfinite(AP).all(); BU = un(UNI.transpose(1, 0, 2).reshape(8 * N, D_)); del UNI
    Qf = {n: un(np.load(f'{FD}/{n}_f.npy').astype(np.float32)) for n in names}; Qz = {n: np.load(f'{FD}/{n}_z.npy') for n in names}; assert all(np.isfinite(Qz[n]).all() for n in names)
    qi = torch.from_numpy(np.sort(np.random.RandomState(0).choice(N, min(5000, N), replace=False))).to(dev); OUT = {}; D = {}
    def terms(g, ru, ro, lu, lo):
        mu = ru.mean(0); d = float(-1 / np.polyfit(lR[:5], mu[:5], 1)[0]); mq = (ru - ro).mean(0); mj = (2 * ru - ro).mean(0); D[g] = d
        for n in names: OUT[f'J{g}|{n}'] = d * ((2 * lu[n] - lo[n]) - mj).min(1); OUT[f'T{g}|{n}'] = d * (lu[n] - mu).min(1); OUT[f'Q{g}|{n}'] = d * ((lu[n] - lo[n]) - mq).min(1)
    Bu_, Bo_ = BU[:N], BU[N:]
    terms('f', ldist(Bu_[qi], Bu_, RS, own=qi, cos=True), ldist(Bu_[qi], Bo_, RO, own=qi, per=7, cos=True), {n: ldist(Qf[n], Bu_, RS, cos=True) for n in names}, {n: ldist(Qf[n], Bo_, RO, cos=True) for n in names})
    del BU, Bu_, Bo_; torch.cuda.empty_cache(); mu_p, sd_p = AP[:, 0].mean(0), AP[:, 0].std(0) + 1e-6
    for e, k in (('X', ix + 1), ('A', ia + 1), ('U', iu + 1)):
        emb = lambda a, k=k: t_((a[..., :k] - mu_p[:k]) / sd_p[:k]); EU = emb(AP[:, 0]); EO = emb(AP[:, 1:].transpose(1, 0, 2).reshape(7 * N, -1))
        terms(f'z{e}', ldist(EU[qi], EU, RS, own=qi), ldist(EU[qi], EO, RO, own=qi, per=7), {n: ldist(emb(Qz[n]), EU, RS) for n in names}, {n: ldist(emb(Qz[n]), EO, RO) for n in names}); del EU, EO; torch.cuda.empty_cache()
    g = lambda q, n: OUT[f'{q}|{n}']
    S = {n: {E2: g('Jf', n) + g('JzX', n), E1: g('Jf', n) + g('JzU', n), Ea: g('Jf', n) + g('JzA', n), P[0]: g('Jf', n), P[1]: g('JzX', n), P[2]: g('Tf', n) + g('JzX', n), P[3]: g('Tf', n) + g('TzX', n), P[4]: g('Tf', n),
             P[5]: g('Tf', n) + g('Qf', n) + g('TzX', n) + g('QzX', n)} for n in names}
    log(tag, f's{seed}', f'bank {N} x 8 | U {iu + 1}, A {ia + 1}, X {ix + 1} of {M["n_blocks"]} | d: ' + ', '.join(f'{k} {v:.1f}' for k, v in D.items()) + f' | identity view vs the TC3 bank: {chk:.2e}', f'{time.time() - t0:.0f}s')
    return S, dict(M, d=D, check=chk)
for tag in TAGS:
    SS = {}; META = {}
    for sd in SEEDS: SS[sd], META[sd] = terms_of(tag, sd)
    names = META[SEEDS[0]]['sets']; idn = 'te-id'; ood = [n for n in names if n.startswith('te-ood-')]; vf, hf = 'te-id__vflip', 'te-id__hflip'; assert vf in names and hf in names
    A = lambda r, n, I=None, J=None: float(np.mean([auc(SS[sd][idn][r][I] if I is not None else SS[sd][idn][r], SS[sd][n][r][J] if J is not None else SS[sd][n][r]) for sd in SEEDS]))
    R = {r: dict(pairs={n[7:]: A(r, n) for n in ood}, vflip=A(r, vf), hflip=A(r, hf), fpr95=float(np.mean([fpr95(SS[sd][idn][r], SS[sd][n][r]) for sd in SEEDS for n in ood]))) for r in ROWS}
    for r in ROWS: R[r]['nat'] = float(np.mean(list(R[r]['pairs'].values())))
    rng = np.random.RandomState(0); nid = len(SS[SEEDS[0]][idn][E2]); B = {k: [] for k in ('E2', 'E2-Tf', 'E2-TfTz', 'E2-E1', 'E2 vflip')}
    for _ in range(2000):
        I = rng.randint(0, nid, nid); Js = {n: rng.randint(0, len(SS[SEEDS[0]][n][E2]), len(SS[SEEDS[0]][n][E2])) for n in ood}; m = lambda r: float(np.mean([A(r, n, I, Js[n]) for n in ood])); e2 = m(E2)
        B['E2'].append(e2); B['E2-Tf'].append(e2 - m(P[4])); B['E2-TfTz'].append(e2 - m(P[3])); B['E2-E1'].append(e2 - m(E1)); B['E2 vflip'].append(A(E2, vf, I, rng.randint(0, nid, nid)))
    CI = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in B.items()}; CTX = {}
    if tag == 'lsb256':                              # stored CFS scores of PROTOCOL_LSUN.md on the same images (higher = more in-distribution), the same auc()
        for q in ('diag', 'knn'):
            v = []
            for sd in SEEDS:
                z = lambda n: np.load(f'{ROOT}/runs/cfs_l7/s{sd}/lsb256/lt-{n}__scores.npz'); zi = z('bedroom'); assert len(zi[q]) == nid
                v.append(dict(nat=float(np.mean([auc(zi[q], z(n[7:])[q]) for n in ood])), vflip=auc(zi[q], z('bedroom__vflip')[q]), hflip=auc(zi[q], z('bedroom__hflip')[q])))
            CTX[f'CFS ({q})'] = {k: float(np.mean([x[k] for x in v])) for k in v[0]}
    RES[tag] = dict(name=NAME[tag], seeds=SEEDS, n_id=nid, n_ood={n[7:]: len(SS[SEEDS[0]][n][E2]) for n in ood}, meta={str(sd): {k: META[sd][k] for k in ('upright', 'i_upright', 'res_upright', 'A', 'i_A', 'exit', 'i_exit', 'n_blocks', 'C', 'd', 'check')} for sd in SEEDS}, rows=R, ci=CI, context=CTX)
    log(f'\n== {NAME[tag]} | seeds {SEEDS} | ID {nid}, OOD ' + ', '.join(f'{k} {v}' for k, v in RES[tag]['n_ood'].items()))
    log(f'{"":50s}{"natural":>8s}{"up-down":>8s}{"mirror":>8s}{"FPR95":>7s}   ' + ' '.join(f'{n[7:][:10]:>10s}' for n in ood))
    for r in ROWS: log(f'{r:50s}{R[r]["nat"]:8.2f}{R[r]["vflip"]:8.2f}{R[r]["hflip"]:8.2f}{R[r]["fpr95"]:7.1f}   ' + ' '.join(f'{R[r]["pairs"][n[7:]]:10.2f}' for n in ood))
    for k, v in CTX.items(): log(f'{"    context: " + k:50s}{v["nat"]:8.2f}{v["vflip"]:8.2f}{v["hflip"]:8.2f}')
    log('95% intervals (paired bootstrap, 2,000 draws): ' + ' | '.join(f'{k} [{v[0]:.2f}, {v[1]:.2f}]' for k, v in CI.items()))
if not DRY: json.dump(RES, open(FOUT, 'w'), indent=1)
log('SX_G_EVAL_DONE', SEEDS, 'dry' if DRY else FOUT, f'{time.time() - t0:.0f}s')
