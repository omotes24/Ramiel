# knorm_depth.py -- the kNN distance of EVERY block (current form: [channel mean, channel std] of one block, unit length, cosine, single
#   precision; nested 10k bank), noise seed 0, for: the val1 sets (runs/up/feat: devall-, sel-), 5,000 held-out train images in the eight
#   orientations of the square (runs/up/feat views.npy; the image's own bank entry is excluded), and the test2 sets (runs/knorm/oracle/feat;
#   ORACLE material, used for development only).  Also the intrinsic dimension d_b of every block (Levina-Bickel, as the kNN term).
#   -> runs/knorm/depth/<tag>_s0.npz : d [nB], rv [nB, 5000, 8, nR], r__<set> [nB, N, nR];  <tag>_s0.json : blocks, parts, res, C, RS, sets
#   usage: python code/knorm_depth.py <gpu> <tag>
import os, sys, json, time, fcntl
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import NPY
from ct_extract_defs import SPEC, bank_indices
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
ROOT = os.path.expanduser('~/ramiel'); dev = torch.device(f'cuda:{int(sys.argv[1])}'); TAG = sys.argv[2]
OD = f'{ROOT}/runs/knorm/depth'; os.makedirs(OD, exist_ok=True)
RS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]; RI = [R - 1 for R in RS]
fd = f'{ROOT}/runs/up/feat/s0/{TAG}'; od2 = f'{ROOT}/runs/knorm/oracle/feat/s0/{TAG}'
M = json.load(open(f'{fd}/meta.json')); BL, Cs, offs, parts, ress = M['blocks'], M['C'], M['off'], M['part'], M['res']; nB = len(BL)
cols = np.concatenate([np.arange(offs[b], offs[b] + 2 * Cs[b]) for b in range(nB)]); coff = np.cumsum([0] + [2 * c for c in Cs])
log = lambda *s: print(*s, flush=True); t0 = time.time()


def load_ms(fn):                      # [N, D] or [N, 8, D] float16 with 7C per block -> the [mean, std] columns, sequential reads
    with open(fn, 'rb') as f:
        ver = np.lib.format.read_magic(f); shape, fo, dt = (np.lib.format.read_array_header_1_0 if ver[0] == 1 else np.lib.format.read_array_header_2_0)(f)
        assert dt == np.float16 and not fo; N = shape[0]; V = shape[1] if len(shape) == 3 else 1; D = shape[-1]
        out = np.empty((N, V, len(cols)), np.float16); step = max(1, 400 // V)
        for i in range(0, N, step):
            k = min(step, N - i); out[i:i + k] = np.fromfile(f, dtype=np.float16, count=k * V * D).reshape(k, V, D)[:, :, cols]
    return out if len(shape) == 3 else out[:, 0]


names1 = sorted(f[:-4] for f in os.listdir(fd) if f.endswith('.npy') and not f.endswith('__idx.npy') and f.startswith(('devall-', 'sel-')))
names2 = sorted(f[:-4] for f in os.listdir(od2) if f.endswith('.npy') and not f.endswith('__idx.npy') and not f.endswith('.tmp.npy'))
with open('/tmp/knorm2_load.lock', 'w') as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    BANK = load_ms(f'{fd}/bank.npy'); Z = {n: load_ms(f'{fd}/{n}.npy') for n in names1}; VW = load_ms(f'{fd}/views.npy')
    fcntl.flock(lk, fcntl.LOCK_UN)
for n in names2: Z[n] = np.load(f'{od2}/{n}.npy'); assert Z[n].shape[1] == len(cols)
log(TAG, 'loaded', BANK.shape, VW.shape, len(names1), len(names2), f'{time.time() - t0:.0f}s')
# the bank entry of each held-out image (if it is in the nested bank)
full = np.load(f'{NPY}/{SPEC[TAG][1]}.npy', mmap_mode='r'); bidx, small = bank_indices(len(full)); bank_tr = np.load(f'{fd}/bank__idx.npy'); pos = np.load(f'{fd}/views__idx.npy')
assert np.array_equal(bank_tr, bidx[small]); where = {int(v): i for i, v in enumerate(bank_tr)}
own = torch.tensor([where.get(int(bidx[p]), -1) for p in pos], device=dev); log(TAG, 'held-out images in the bank:', int((own >= 0).sum()), 'of', len(pos))


def knn(Q, B, own=None):
    out = []; k = min(max(RS), len(B) - (1 if own is not None else 0))
    for i in range(0, len(Q), 1024):
        g = Q[i:i + 1024] @ B.T
        if own is not None:
            o = own[i:i + 1024]; m = torch.nonzero(o >= 0).squeeze(1); g[m, o[m]] = -2.0
        out.append(torch.sqrt((2 * (1 - g.topk(k, dim=1).values)).clamp_min(2e-9))[:, RI])
    return torch.cat(out).cpu().numpy().astype(np.float32)


def dim_lb(B):
    idx = torch.from_numpy(np.random.RandomState(0).choice(len(B), min(3000, len(B)), replace=False)).to(dev)
    r = torch.sqrt((2 * (1 - (B[idx] @ B.T).topk(21, dim=1).values[:, 1:])).clamp_min(2e-6))
    dd = 1 / torch.log(r[:, -1:] / r[:, :-1]).mean(1); fin = torch.isfinite(dd); return float(dd[fin].mean())


unit = lambda a: F.normalize(torch.from_numpy(np.ascontiguousarray(a)).to(dev).float(), dim=1)
D_, RV, RR = np.zeros(nB), np.zeros((nB, VW.shape[0], 8, len(RS)), np.float32), {n: [] for n in Z}
for b in range(nB):
    sl = slice(coff[b], coff[b + 1]); B = unit(BANK[:, sl]); D_[b] = dim_lb(B)
    for v in range(8): RV[b, :, v] = knn(unit(VW[:, v, sl]), B, own)
    for n in Z: RR[n].append(knn(unit(Z[n][:, sl]), B))
    log(TAG, f'b{b:02d} {parts[b][0]}{ress[b]} C{Cs[b]} d {D_[b]:.1f}', f'{time.time() - t0:.0f}s')
np.savez(f'{OD}/{TAG}_s0.npz', d=D_, rv=RV, **{f'r__{n}': np.stack(v) for n, v in RR.items()})
json.dump(dict(tag=TAG, blocks=BL, part=parts, res=ress, C=Cs, RS=RS, val1=names1, test2=names2, ori=['id', 'rot90', 'rot180', 'rot270', 'hflip', 'vflip', 'transp', 'atransp']),
          open(f'{OD}/{TAG}_s0.json', 'w'))
log('KNORM_DEPTH_DONE', TAG, f'{time.time() - t0:.0f}s')
