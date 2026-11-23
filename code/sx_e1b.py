# sx_e1b.py -- PROTOCOL_SX.md, addendum F: the benchmark of CFS (E1 of PROTOCOL_AX.md: ONE CIFAR-10 EDM backbone for ID in {cifar10, svhn,
#   celeba}; OOD = the other two, CIFAR-100 and DTD; full test splits) for the detector of addendum E: ONE forward pass, NO training.
#   The upright block U is the stored one of code/ax_e1.py (chosen from the train images of the ID dataset; seed 0 rule);
#   X = the last decoder block at twice the resolution of U (A = the first one).  One pass stopped at X gives f (unit [mean, std] of U)
#   and z (amplitude profile of the blocks 1..X) for the 50,000 bank images in eight orientations (v = 4 f + r, D_v x = R_r(H^f x), as
#   ax_e1.py; the image is transformed, the noise is not) and for every test image.  Terms exactly as code/sx_e_terms.py:
#     J_g = d_g * min_R{(2 lu_R - lo_R) - mean},   score E2 = J_f + J_z   (E1 / Ea: z of the blocks 1..U / 1..A; T, Q: the two parts)
#   usage: python code/sx_e1b.py --id svhn --gpu 0 --seed 0     -> runs/sx/e1b/<id>_s<seed>.npz (terms per set), <id>_s<seed>.json
import os, sys, json, time, argparse
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import ROOT, NPY, load_model, to_t, TF, shared_noise
from ct_extract_defs import SPEC, SIGS, bank_indices
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
p = argparse.ArgumentParser(); p.add_argument('--id', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--seed', type=int, default=0); p.add_argument('--bs', type=int, default=500)
a = p.parse_args(); dev = torch.device(f'cuda:{a.gpu}'); log = lambda *s: print(*s, flush=True); t0 = time.time(); sig = SIGS[0]; assert abs(sig - 0.01) < 1e-12
TRAIN = {'cifar10': 'cifar10_train', 'svhn': 'svhn_train', 'celeba': 'celeba_train'}
POOL = {'cifar10': 'cifar10_test', 'svhn': 'svhn_test', 'celeba': 'celeba_test', 'cifar100': 'cifar100_test', 'dtd': 'textures'}
OD = f'{ROOT}/runs/sx/e1b'; os.makedirs(OD, exist_ok=True); FO = f'{OD}/{a.id}_s{a.seed}'; assert not os.path.exists(FO + '.json'), 'done already'
S = json.load(open(f'{ROOT}/runs/ax/e1/{a.id}_s{a.seed}.json')); KZ = np.load(f'{ROOT}/runs/ax/e1/{a.id}_s{a.seed}.npz'); assert S['model'] == 'edmc10'; bu = S['block']
M = json.load(open(f'{ROOT}/runs/up/feat/s0/edmc10/meta.json')); BL, parts, ress = M['blocks'], M['part'], M['res']; nB = len(BL); D = 2 * M['C'][bu]
cand = [b for b in range(nB) if parts[b] == 'dec' and ress[b] == 2 * ress[bu]] or [nB - 1]; ba, bx = cand[0], cand[-1]; nb = bx + 1; assert bu < nb
den, net = load_model(SPEC['edmc10'][0], dev); mods = dict(net.named_modules()); store = {}
class Stop(Exception): pass
def mk(i):
    def f(m, inp, o):
        o = (o[0] if isinstance(o, (tuple, list)) else o).float(); store[i] = (o - o.mean((2, 3), keepdim=True)).abs().amax((2, 3)).clamp_min(1e-20).log().mean(1)
        if i == bu: store['f'] = torch.cat([o.mean((2, 3)), o.std((2, 3), correction=0)], 1)
        if i == bx: raise Stop()
    return f
for i in range(nb): mods[BL[i]].register_forward_hook(mk(i))
ROT = ['id', 'rot90', 'rot180', 'rot270']
def view(x, v): f, r = divmod(v, 4); return TF[ROT[r]][0](x.flip(3) if f else x)
@torch.no_grad()
def desc(x):
    eps = shared_noise(x.shape[1:], a.seed, 0, 0, dev); store.clear()
    try: den(x + sig * eps, sig)
    except Stop: pass
    return store['f'].half().cpu().numpy(), torch.stack([store[i] for i in range(nb)], 1).cpu().numpy()
full = np.load(f'{NPY}/{TRAIN[a.id]}.npy', mmap_mode='r'); bidx, small = bank_indices(len(full)); N = len(bidx); assert N == S['n_bank']
FB = np.empty((N, 8, D), np.float16); ZB = np.empty((N, 8, nb), np.float32)
for i in range(0, N, a.bs):
    x = to_t(np.ascontiguousarray(np.stack([full[j] for j in bidx[i:i + a.bs]])), dev)
    for v in range(8): FB[i:i + len(x), v], ZB[i:i + len(x), v] = desc(view(x, v))
log(a.id, a.seed, f'bank {N} x 8 | U {S["label"]} ({bu + 1}), A {ba + 1}, X {bx + 1} of {nB} | f {D}, z {nb}', f'{time.time() - t0:.0f}s')
names = list(POOL); QF, QZ, NS = {}, {}, {}
for nm in names:
    xs = np.load(f'{NPY}/{POOL[nm]}.npy', mmap_mode='r'); n = len(xs); qf = np.empty((n, D), np.float16); qz = np.empty((n, nb), np.float32)
    for i in range(0, n, a.bs): qf[i:i + a.bs], qz[i:i + a.bs] = desc(to_t(np.ascontiguousarray(xs[i:i + a.bs]), dev))
    QF[nm], QZ[nm], NS[nm] = qf, qz, int(n); assert n == len(KZ[f'K|{nm}']); log(a.id, a.seed, 'set', nm, n, f'{time.time() - t0:.0f}s')
del den, net; torch.cuda.empty_cache()
# ---- terms: the code of code/sx_e_terms.py
RS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]; RO = [7 * r for r in RS]; lR = np.log(RS)
t_ = lambda q: torch.from_numpy(np.ascontiguousarray(q)).to(dev).float(); un = lambda q: F.normalize(t_(q), dim=-1)
def ldist(Q, Bk, ranks, own=None, per=1, cos=False, bs=256):
    out = []; n2 = None if cos else (Bk * Bk).sum(1); nimg = len(Bk) // per
    for i in range(0, len(Q), bs):
        q = Q[i:i + bs]; d2 = 2 - 2 * q @ Bk.T if cos else (q * q).sum(1, keepdim=True) + n2[None] - 2 * q @ Bk.T
        if own is not None: d2.view(len(q), per, nimg)[torch.arange(len(q), device=dev), :, own[i:i + bs]] = float('inf')
        out.append(-0.5 * torch.log((-(-d2).topk(max(ranks), dim=1).values[:, [r - 1 for r in ranks]]).clamp_min(1e-12)))
    return torch.cat(out).double().cpu().numpy()
qi = torch.from_numpy(np.sort(np.random.RandomState(0).choice(N, min(5000, N), replace=False))).to(dev); OUT = {}; DIM = {}
def terms(g, ru, ro, lu, lo):
    mu = ru.mean(0); d = float(-1 / np.polyfit(lR[:5], mu[:5], 1)[0]); mq = (ru - ro).mean(0); mj = (2 * ru - ro).mean(0); DIM[g] = d
    for n in names: OUT[f'J{g}|{n}'] = d * ((2 * lu[n] - lo[n]) - mj).min(1); OUT[f'T{g}|{n}'] = d * (lu[n] - mu).min(1); OUT[f'Q{g}|{n}'] = d * ((lu[n] - lo[n]) - mq).min(1)
BU = un(FB.transpose(1, 0, 2).reshape(8 * N, D)); Bu_, Bo_ = BU[:N], BU[N:]
terms('f', ldist(Bu_[qi], Bu_, RS, own=qi, cos=True), ldist(Bu_[qi], Bo_, RO, own=qi, per=7, cos=True), {n: ldist(un(QF[n]), Bu_, RS, cos=True) for n in names}, {n: ldist(un(QF[n]), Bo_, RO, cos=True) for n in names})
del BU, Bu_, Bo_; torch.cuda.empty_cache(); mu_p, sd_p = ZB[:, 0].mean(0), ZB[:, 0].std(0) + 1e-6
for e, k in (('X', bx + 1), ('A', ba + 1), ('U', bu + 1)):
    emb = lambda q, k=k: t_((q[..., :k] - mu_p[:k]) / sd_p[:k]); EU = emb(ZB[:, 0]); EO = emb(ZB[:, 1:].transpose(1, 0, 2).reshape(7 * N, -1))
    terms(f'z{e}', ldist(EU[qi], EU, RS, own=qi), ldist(EU[qi], EO, RO, own=qi, per=7), {n: ldist(emb(QZ[n]), EU, RS) for n in names}, {n: ldist(emb(QZ[n]), EO, RO) for n in names}); del EU, EO; torch.cuda.empty_cache()
def auc(x, y): r = rankdata(np.concatenate([x, y])); return 100 * float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))
g = lambda ks: (lambda nm: sum(OUT[f'{k}|{nm}'] for k in ks)); kz = lambda ks: (lambda nm: sum(KZ[f'{k}|{nm}'].astype(np.float64) for k in ks))
ROWS = {'E2': g(('Jf', 'JzX')), 'E1': g(('Jf', 'JzU')), 'Ea': g(('Jf', 'JzA')), 'Jf': g(('Jf',)), 'JzX': g(('JzX',)), 'Tf': g(('Tf',)), 'Tf+JzX': g(('Tf', 'JzX')), 'four': g(('Tf', 'Qf', 'TzX', 'QzX')),
        'stored: trained detector of 1122d': kz('KCM'), 'stored: its kNN term': kz('K')}
RESULT = {row: {o: auc(f(a.id), f(o)) for o in names if o != a.id} for row, f in ROWS.items()}
for row in ROWS: log(a.id, a.seed, f'{row:34s}', ' '.join(f'{o} {v:.2f}' for o, v in RESULT[row].items()), '| mean', f'{np.mean(list(RESULT[row].values())):.2f}')
np.savez(FO + '.npz', **OUT); json.dump(dict(id=a.id, seed=a.seed, block=bu, label=S['label'], i_A=ba, i_exit=bx, n_blocks=nB, n_bank=N, n_sets=NS, d=DIM, auroc=RESULT), open(FO + '.json', 'w'), indent=1)
log('SX_E1B_DONE', a.id, a.seed, f'{time.time() - t0:.0f}s')
