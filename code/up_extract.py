# up_extract.py -- per-channel statistics of EVERY encoder / decoder block from ONE pass of the frozen denoiser at sigma_1 = 0.01
#   (x + sigma_1 eps; eps = the shared noise image of the seed, as Upright kNN).  Development tool for the question
#   "where in the forward pass can one read whether an image is upright?"  (PROTOCOL_UP.md).
#   Per block with C channels: m = spatial mean, s = spatial std, a = log max_u |f - m| (log-amplitude), qm = 2x2-grid means
#   (4C; TL, TR, BL, BR)  ->  7C values, float16, blocks concatenated in forward order (offsets in meta.json).
#   views : nh bank images x the 8 orientations of the square; the image is transformed, the noise is NOT (V_g = g(x) + sigma_1 eps)
#           images = perm[:nhold] (inside Upright kNN's 5,000 held-out bank images) + perm[5000:5000+nh-nhold], perm = RandomState(0)
#           -> views.npy [nh, 8, D], views__idx.npy (positions in the bank), order of ORI below
#   bank  : the nested small bank (10k) upright -> bank.npy [nb, D]
#   sets  : every set of runs/ok/sets_<tag>.json (and --sets) whose name starts with one of --prefixes, as stored (pre applied) -> <set>.npy
#   usage: python code/up_extract.py --tag edmc10 --gpu 1 [--seed 0] [--nh 5000] [--prefixes devall-,sel-,fresh-]
import os, sys, re, json, time, argparse
import numpy as np
import torch
from numpy.lib.format import open_memmap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dgcore
from dgcore import ROOT, NPY, to_t, TF, shared_noise
from ct_extract_defs import SPEC, SIGS, PATS, part, bank_indices

p = argparse.ArgumentParser()
p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--seed', type=int, default=0)
p.add_argument('--nh', type=int, default=5000); p.add_argument('--nhold', type=int, default=1000); p.add_argument('--k', type=int, default=0)
p.add_argument('--prefixes', default='devall-,sel-,fresh-'); p.add_argument('--sets', default=''); p.add_argument('--out', default='runs/up/feat')
p.add_argument('--bank', type=int, default=1); p.add_argument('--views', type=int, default=1)
a = p.parse_args(); PRE = tuple(q for q in a.prefixes.split(',') if q)
spec, bankset, bs = SPEC[a.tag]; sig = SIGS[a.k]
dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True
log = lambda *s: print(*s, flush=True)
den, net = dgcore.load_model(spec, dev); mods = dict(net.named_modules())
cand = [n for n in mods if any(re.search(q, n) for q in PATS)]; order, res, ch = [], {}, {}; store = {}


def grab(name):
    def f(m, inp, o):
        if isinstance(o, (tuple, list)): o = o[0]
        if torch.is_tensor(o) and o.ndim == 4:
            if name not in res: order.append(name)
            res[name] = int(o.shape[-1]); ch[name] = int(o.shape[1]); store[name] = o
    return f


for n in cand: mods[n].register_forward_hook(grab(n))
full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r')
with torch.no_grad(): den(to_t(np.ascontiguousarray(full[:2]), dev), 0.2)
BLOCKS = list(order); off = np.cumsum([0] + [7 * ch[n] for n in BLOCKS]); D = int(off[-1])
od = f'{ROOT}/{a.out}/s{a.seed}/{a.tag}'; os.makedirs(od, exist_ok=True)
json.dump(dict(tag=a.tag, spec=spec, sigma=sig, seed=a.seed, blocks=BLOCKS, C=[ch[n] for n in BLOCKS], res=[res[n] for n in BLOCKS],
               part=[part(n) for n in BLOCKS], off=[int(v) for v in off], D=D, layout='m,s,a,qm(TL,TR,BL,BR)'), open(f'{od}/meta.json', 'w'), indent=1)
log(a.tag, len(BLOCKS), 'blocks, D =', D, '| sigma', sig)
ORI = ['id', 'rot90', 'rot180', 'rot270', 'hflip', 'vflip', 'transp', 'atransp']


def pool(o):
    o = o.float(); B, C, H, W = o.shape; m = o.mean((2, 3)); s = o.std((2, 3), correction=0)
    amp = (o - m[..., None, None]).abs().amax((2, 3)).clamp_min(1e-20).log()
    qm = o.reshape(B, C, 2, H // 2, 2, W // 2).mean((3, 5)).permute(0, 2, 3, 1).reshape(B, 4 * C) if H >= 2 and W >= 2 else m.repeat(1, 4)
    return torch.cat([m, s, amp, qm], 1)


@torch.no_grad()
def feats(x):
    eps = shared_noise(x.shape[1:], a.seed, a.k, 0, dev); store.clear(); den(x + sig * eps, sig)
    return torch.cat([pool(store[n]) for n in BLOCKS], 1).half().cpu().numpy()


def run(fn, xu, pres):
    if os.path.exists(fn): log('exists', fn); return
    N = len(xu); M = open_memmap(fn + '.part', 'w+', np.float16, (N, len(pres), D) if len(pres) > 1 else (N, D))
    for i in range(0, N, bs):
        x = to_t(np.ascontiguousarray(xu[i:i + bs]), dev)
        for v, pre in enumerate(pres):
            f = feats(x if pre in ('none', 'id') else TF[pre][0](x))
            if len(pres) > 1: M[i:i + len(x), v] = f
            else: M[i:i + len(x)] = f
    M.flush(); del M; os.replace(fn + '.part', fn)


t0 = time.time(); bidx, small = bank_indices(len(full))
if a.views:
    perm = np.random.RandomState(0).permutation(len(bidx)); pos = np.concatenate([perm[:a.nhold], perm[5000:5000 + a.nh - a.nhold]])
    np.save(f'{od}/views__idx.npy', pos); run(f'{od}/views.npy', np.stack([full[j] for j in bidx[pos]]), ORI)
    log(a.tag, 'views', len(pos), f'{time.time() - t0:.0f}s')
if a.bank:
    np.save(f'{od}/bank__idx.npy', bidx[small]); run(f'{od}/bank.npy', np.stack([full[j] for j in bidx[small]]), ['none'])
    log(a.tag, 'bank', len(small), f'{time.time() - t0:.0f}s')
ALL = {}
for f_ in [f'runs/ok/sets_{a.tag}.json'] + ([a.sets] if a.sets else []):
    for r in json.load(open(f'{ROOT}/{f_}')): ALL.setdefault(r[0], r)
for name, data, idx, pre in ALL.values():
    if not name.startswith(PRE): continue
    path = f'{NPY}/{data}.npy'
    if not os.path.exists(path): log('SKIP', name, 'missing', path); continue
    idx = np.asarray(idx, np.int64); xs = np.load(path, mmap_mode='r')
    run(f'{od}/{name}.npy', np.stack([xs[j] for j in idx]), [pre]); np.save(f'{od}/{name}__idx.npy', idx)
    log(a.tag, name, len(idx), f'{time.time() - t0:.0f}s')
log('UP_EXTRACT_DONE', a.tag, f's{a.seed}', f'{time.time() - t0:.0f}s')
