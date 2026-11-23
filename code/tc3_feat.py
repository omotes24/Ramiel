# tc3_feat.py -- PROTOCOL_TC3.md: [channel mean, channel std] of EVERY block from ONE pass of the frozen denoiser at sigma_1 = 0.01
#   (x + sigma_1 eps; eps = the shared noise image of the seed; the image is transformed, the noise is NOT), for a new tier.
#   The selection of blocks, of the bank and of the held-out views is that of up_extract.py (the features of the upright-block rule):
#     blocks : every module matching ct_extract_defs.PATS with a 4-D output, in forward order
#     bank   : the nested small bank (10k) upright                                   -> bank.npy [nb, D], bank__idx.npy
#     views  : 5,000 bank images x the 8 orientations of the square; images = perm[:1000] + perm[5000:9000], perm = RandomState(0)
#              -> views_p<i>of<n>.npy [n_i, 8, D] (the i-th of n contiguous parts), views__idx.npy (positions in the bank)
#     sets   : every row of runs/tc3/sets_<tag>.json (pre applied)                    -> <name>.npy [N, D], <name>__idx.npy
#   D = sum_b 2 C_b, float16.  Models: ADM 256x256 (guided-diffusion) through admwrap.load_adm, fp16, sigma_1 = sigma_t at t = 0.
#   usage: python code/tc3_feat.py --tag lsb256 --gpu 0 --what views --part 0/2 | --what bank | --what sets   [--limit 16 --out runs/tc3_smoke/feat]
import os, sys, re, json, time, argparse
import numpy as np, torch
from numpy.lib.format import open_memmap
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dgcore, admwrap
from dgcore import ROOT, NPY, to_t, TF, shared_noise
from ct_extract_defs import SIGS, PATS, part, bank_indices
TIERS = {'lsb256': ('lsun_bedroom.pt', 'l256_bedroom_bank', 8), 'adm256c': ('256x256_diffusion_uncond.pt', 'j256_imagenet_train10', 8)}
p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--gpu', type=int, default=0); p.add_argument('--seed', type=int, default=0)
p.add_argument('--what', required=True, choices=['bank', 'views', 'sets']); p.add_argument('--part', default='0/1'); p.add_argument('--limit', type=int, default=0)
p.add_argument('--out', default='runs/tc3/feat')
a = p.parse_args(); ck, bankset, bs = TIERS[a.tag]; sig = SIGS[0]; assert abs(sig - 0.01) < 1e-12
dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True; log = lambda *s: print(*s, flush=True)
admwrap.CKPT = f'{ROOT}/ckpt/{ck}'; den, net = admwrap.load_adm(dev, fp16=True); mods = dict(net.named_modules())
cand = [n for n in mods if any(re.search(q, n) for q in PATS)]; order, res, ch, store = [], {}, {}, {}


def grab(name):
    def f(m, inp, o):
        if isinstance(o, (tuple, list)): o = o[0]
        if torch.is_tensor(o) and o.ndim == 4:
            if name not in res: order.append(name)
            res[name] = int(o.shape[-1]); ch[name] = int(o.shape[1]); of = o.float()
            store[name] = torch.cat([of.mean((2, 3)), of.std((2, 3), correction=0)], 1)
    return f


for n in cand: mods[n].register_forward_hook(grab(n))
full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r')
with torch.no_grad(): den(to_t(np.ascontiguousarray(full[:2]), dev), sig)
BLOCKS = list(order); C = [ch[n] for n in BLOCKS]; off = np.cumsum([0] + [2 * c for c in C]); D = int(off[-1])
od = f'{ROOT}/{a.out}/s{a.seed}/{a.tag}'; os.makedirs(od, exist_ok=True)
meta = dict(tag=a.tag, ckpt=ck, bank=bankset, sigma=sig, seed=a.seed, blocks=BLOCKS, C=C, res=[res[n] for n in BLOCKS], part=[part(n) for n in BLOCKS], off=[int(v) for v in off], D=D, layout='m,s')
if os.path.exists(f'{od}/meta.json'): assert json.load(open(f'{od}/meta.json')) == meta, 'meta differs'
else:
    json.dump(meta, open(f'{od}/meta.{os.getpid()}.tmp', 'w'), indent=1); os.replace(f'{od}/meta.{os.getpid()}.tmp', f'{od}/meta.json')
log(a.tag, a.what, a.part, len(BLOCKS), 'blocks, D =', D, '| sigma', sig)
ORI = ['id', 'rot90', 'rot180', 'rot270', 'hflip', 'vflip', 'transp', 'atransp']


@torch.no_grad()
def feats(x):
    eps = shared_noise(x.shape[1:], a.seed, 0, 0, dev); store.clear(); den(x + sig * eps, sig)
    return torch.cat([store[n] for n in BLOCKS], 1).half().cpu().numpy()


def run(fn, xu, pres):
    if os.path.exists(fn): log('exists', fn); return
    N = len(xu); M = open_memmap(fn + '.part', 'w+', np.float16, (N, len(pres), D) if len(pres) > 1 else (N, D)); t1 = time.time()
    for i in range(0, N, bs):
        x = to_t(np.ascontiguousarray(xu[i:i + bs]), dev)
        for v, pre in enumerate(pres):
            f = feats(x if pre in ('none', 'id') else TF[pre][0](x))
            if len(pres) > 1: M[i:i + len(x), v] = f
            else: M[i:i + len(x)] = f
        if i % (bs * 50) == 0: log('  ', os.path.basename(fn), i, N, f'{time.time() - t1:.0f}s')
    M.flush(); del M; os.replace(fn + '.part', fn)


t0 = time.time(); bidx, small = bank_indices(len(full)); lim = (lambda v: v[:a.limit]) if a.limit else (lambda v: v)
if a.what == 'views':
    perm = np.random.RandomState(0).permutation(len(bidx)); pos = np.concatenate([perm[:1000], perm[5000:9000]]); assert len(pos) == 5000
    i, n = [int(v) for v in a.part.split('/')]; sel = lim(np.array_split(np.arange(len(pos)), n)[i])
    np.save(f'{od}/views__idx.npy', pos); run(f'{od}/views_p{i}of{n}.npy', np.stack([full[j] for j in bidx[pos[sel]]]), ORI)
    log(a.tag, 'views part', a.part, len(sel), f'{time.time() - t0:.0f}s')
elif a.what == 'bank':
    sm = lim(small); np.save(f'{od}/bank__idx.npy', bidx[sm]); run(f'{od}/bank.npy', np.stack([full[j] for j in bidx[sm]]), ['none'])
    log(a.tag, 'bank', len(sm), f'{time.time() - t0:.0f}s')
else:
    for name, data, idx, pre in json.load(open(f'{ROOT}/runs/tc3/sets_{a.tag}.json')):
        assert pre == 'none' or name.endswith('__' + pre), (name, pre)
        idx = lim(np.asarray(idx, np.int64)); xs = np.load(f'{NPY}/{data}.npy', mmap_mode='r')
        run(f'{od}/{name}.npy', np.stack([xs[j] for j in idx]), [pre]); np.save(f'{od}/{name}__idx.npy', idx)
        log(a.tag, name, len(idx), f'{time.time() - t0:.0f}s')
log('TC3_FEAT_DONE', a.tag, a.what, a.part, f'{time.time() - t0:.0f}s')
