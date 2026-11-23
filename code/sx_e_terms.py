# sx_e_terms.py -- PROTOCOL_SX.md, addendum E: per-image terms of the detector with ONE forward pass and NO training (no AUROC here).
#   One pass of the frozen denoiser (sigma_1, shared noise image), stopped at block B (the upright block U | A | the exit block X) gives
#     f = unit [channel mean, channel std] of the upright block U,
#     z = amplitude profile: one number per block 1..B (channel mean of log max_u |f_h - mean_u f_h|), each block centred and scaled by
#         the mean and the standard deviation of the upright train profiles.
#   Bank: the train images (all 50,000; ImageNet-64 10,000) in the eight orientations; u = as given, o = the seven other orientations.
#   For a descriptor g in {f, z} and the ranks R in {1, 2, 5, ..., 1000}:
#     lu_R = -log r_R(g; u)     r_R  = distance to the R-th nearest upright bank entry (chord distance for f, Euclidean for z)
#     lo_R = -log r_7R(g; o)    the same probability mass in the bank of the seven other orientations
#     J_g = d_g * min_R{(2 lu_R - lo_R) - mean}     =  log p_u + log(p_u / p_o)  by nearest neighbours        (the detector: J_f + J_z)
#     T_g = d_g * min_R{lu_R - mean},   Q_g = d_g * min_R{(lu_R - lo_R) - mean}                                 (the two parts, reported)
#   mean: over 5,000 train images with every copy of their own image left out; d_g = -1 / slope of the mean of lu_R over log R, R = 1..20.
#   Nothing is fitted: no gradient step, no head, no discriminant, no temperature, no rank chosen from labels.
#   usage: python code/sx_e_terms.py <gpu> <tag> <seed> [check]      -> runs/sx/e_final/terms_<tag>_s<seed>.npz, meta_<tag>_s<seed>.json
#          check (seed 0 only): the same code on the val2 sets against runs/sx/amp5/terms_<tag>.npz; no test image is read.
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F
ROOT = os.path.expanduser('~/ramiel'); dev = torch.device(f'cuda:{int(sys.argv[1])}'); TAG = sys.argv[2]; seed = int(sys.argv[3]); CHECK = len(sys.argv) > 4 and sys.argv[4] == 'check'
OD = f'{ROOT}/runs/sx/e_final'; os.makedirs(OD, exist_ok=True); log = lambda *s: print(*s, flush=True); t0 = time.time()
RS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]; RO = [7 * r for r in RS]; lR = np.log(RS)
t_ = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev).float(); un = lambda a: F.normalize(t_(a), dim=-1)
FD = f'{ROOT}/runs/sx/nt/feat/s{seed}/{TAG}'; UNI = np.load(f'{FD}/union.npy'); N, _, D_ = UNI.shape; BU = un(UNI.transpose(1, 0, 2).reshape(8 * N, D_)); del UNI
AP = np.load(f'{FD}/ap/bank.npy'); MA = json.load(open(f'{FD}/ap/meta.json')); iu, ia, ix = MA['i_upright'], MA['i_A'], MA['i_exit']; assert AP.shape[0] == N and AP.shape[1] == 8 and AP.shape[2] == ix + 1
if CHECK:
    assert seed == 0; UP = f'{ROOT}/runs/knorm/up/s0/{TAG}'; fd = f'{ROOT}/runs/up/feat/s0/{TAG}'; M = json.load(open(f'{fd}/meta.json')); off = M['off'][M['blocks'].index(json.load(open(f'{UP}/meta.json'))['name'])]
    names = sorted(f[:-4] for f in os.listdir(fd) if f.endswith('.npy') and not f.endswith('__idx.npy') and f.startswith('fresh-') and os.path.exists(f'{FD}/ap/{f}'))
    Qf = {n: un(np.asarray(np.load(f'{fd}/{n}.npy', mmap_mode='r')[:, off:off + D_])) for n in names}
else:
    names = sorted(f[:-4] for f in os.listdir(f'{FD}/tv') if f.endswith('.npy')); assert all(n.startswith(('test2-', 'testc-')) for n in names) and all(os.path.exists(f'{FD}/ap/{n}.npy') for n in names)
    Qf = {n: un(np.load(f'{FD}/tv/{n}.npy')[:, 0]) for n in names}
Qz = {n: np.load(f'{FD}/ap/{n}.npy') for n in names}; assert all(len(Qz[n]) == len(Qf[n]) and Qz[n].shape[1] == ix + 1 for n in names)
def ldist(Q, Bk, ranks, own=None, per=1, cos=False, bs=256):     # -log distance to the ranks-th nearest row of Bk; own: index of the query's own image (left out in all its `per` copies; Bk stacked copy-major)
    out = []; n2 = None if cos else (Bk * Bk).sum(1); nimg = len(Bk) // per
    for i in range(0, len(Q), bs):
        q = Q[i:i + bs]; d2 = 2 - 2 * q @ Bk.T if cos else (q * q).sum(1, keepdim=True) + n2[None] - 2 * q @ Bk.T
        if own is not None: d2.view(len(q), per, nimg)[torch.arange(len(q), device=dev), :, own[i:i + bs]] = float('inf')
        out.append(-0.5 * torch.log((-(-d2).topk(max(ranks), dim=1).values[:, [r - 1 for r in ranks]]).clamp_min(1e-12)))
    return torch.cat(out).double().cpu().numpy()
qi = torch.from_numpy(np.sort(np.random.RandomState(0).choice(N, min(5000, N), replace=False))).to(dev); RAW = {}; OUT = {}; META = dict(tag=TAG, seed=seed, N=N, n_sets=len(names), i_upright=iu, i_A=ia, i_exit=ix, d={})
def terms(g, ru, ro, lu, lo):                      # ru, ro: [5000, 10] leave-one-out train values; lu, lo: {set: [n, 10]}
    mu = ru.mean(0); d = float(-1 / np.polyfit(lR[:5], mu[:5], 1)[0]); mq = (ru - ro).mean(0); mj = (2 * ru - ro).mean(0); META['d'][g] = d
    OUT[f'ref|J{g}'] = d * ((2 * ru - ro) - mj).min(1); OUT[f'ref|T{g}'] = d * (ru - mu).min(1); OUT[f'ref|Q{g}'] = d * ((ru - ro) - mq).min(1)
    for n in names: OUT[f'J{g}|{n}'] = d * ((2 * lu[n] - lo[n]) - mj).min(1); OUT[f'T{g}|{n}'] = d * (lu[n] - mu).min(1); OUT[f'Q{g}|{n}'] = d * ((lu[n] - lo[n]) - mq).min(1)
    RAW[f'ref|lu{g}'] = ru; RAW[f'ref|lo{g}'] = ro
    for n in names: RAW[f'lu{g}|{n}'] = lu[n]; RAW[f'lo{g}|{n}'] = lo[n]
Bu_, Bo_ = BU[:N], BU[N:]
terms('f', ldist(Bu_[qi], Bu_, RS, own=qi, cos=True), ldist(Bu_[qi], Bo_, RO, own=qi, per=7, cos=True), {n: ldist(Qf[n], Bu_, RS, cos=True) for n in names}, {n: ldist(Qf[n], Bo_, RO, cos=True) for n in names})
log(TAG, seed, f'bank {N} x 8 | f done (d {META["d"]["f"]:.1f})', f'{time.time() - t0:.0f}s'); del BU, Bu_, Bo_; torch.cuda.empty_cache()
mu_p, sd_p = AP[:, 0].mean(0), AP[:, 0].std(0) + 1e-6
for e, k in (('X', ix + 1), ('A', ia + 1), ('U', iu + 1)):
    emb = lambda a, k=k: t_((a[..., :k] - mu_p[:k]) / sd_p[:k]); EU = emb(AP[:, 0]); EO = emb(AP[:, 1:].transpose(1, 0, 2).reshape(7 * N, -1))
    terms(f'z{e}', ldist(EU[qi], EU, RS, own=qi), ldist(EU[qi], EO, RO, own=qi, per=7), {n: ldist(emb(Qz[n]), EU, RS) for n in names}, {n: ldist(emb(Qz[n]), EO, RO) for n in names})
    log(TAG, seed, f'z of blocks 1..{k} ({e}) done (d {META["d"]["z" + e]:.1f})', f'{time.time() - t0:.0f}s'); del EU, EO; torch.cuda.empty_cache()
if CHECK:
    Z = np.load(f'{ROOT}/runs/sx/amp5/terms_{TAG}.npz'); mx = 0.0
    for a, b_ in (('lf', 'luf'), ('lfo', 'lof'), ('lu[X]', 'luzX'), ('lo[X]', 'lozX'), ('lu[A]', 'luzA'), ('lo[A]', 'lozA'), ('lu[U]', 'luzU'), ('lo[U]', 'lozU')):
        mx = max([mx, float(np.abs(Z[f'ref|{a}'] - RAW[f'ref|{b_}']).max())] + [float(np.abs(Z[f'{a}|{n}'] - RAW[f'{b_}|{n}']).max()) for n in names])
    log('SX_E_CHECK', TAG, f'{len(names)} val2 sets: max abs difference of the log distances against runs/sx/amp5: {mx:.2e}', f'{time.time() - t0:.0f}s'); sys.exit(0)
np.savez(f'{OD}/terms_{TAG}_s{seed}.npz', **OUT); np.savez(f'{OD}/raw_{TAG}_s{seed}.npz', **RAW); json.dump(META, open(f'{OD}/meta_{TAG}_s{seed}.json', 'w'), indent=1)
log('SX_E_TERMS_DONE', TAG, seed, f'bank {N} x 8 | d: ' + ', '.join(f'{k} {v:.1f}' for k, v in META['d'].items()) + f' | {len(names)} sets', f'{time.time() - t0:.0f}s')
