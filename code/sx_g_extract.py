# sx_g_extract.py -- PROTOCOL_SX.md, addendum G: the features of the detector of addendum E (ONE truncated forward pass, NO training)
#   on the 256x256 tiers of PROTOCOL_TC3.md (ADM, Dhariwal & Nichol 2021): lsb256 = LSUN bedroom, adm256c = ImageNet unconditional.
#   One pass at sigma_1 = 0.01 with the shared noise image of the seed (the image is transformed, the noise is not), fp16 model,
#   STOPPED at the exit block X = last decoder block at twice the resolution of the upright block U (U = argmax_b A_b of
#   runs/tc3/depth_<tag>.json, chosen from training images only).  Read from the pass:
#     f = [channel mean, channel std] of block U                      -> *_f*.npy   float16
#     a = amplitude profile, one number per block 1..X:  channel mean of log max_u |h - mean_u h|   -> *_z*.npy   float32
#   bank: the 10,000 bank images in the eight orientations of the square (id, rot90, rot180, rot270, hflip, vflip, transp, atransp)
#   sets: the te- rows of runs/tc3/sets_<tag>.json (test ID, its upside-down and mirrored copies, the test OOD sets)
#   output: runs/sx/g/feat/s<seed>/<tag>/{bank_f_p<i>of<n>.npy [n_i, 8, 2C], bank_z_p<i>of<n>.npy [n_i, 8, nb], <set>_f.npy, <set>_z.npy, meta.json}
#   usage: python code/sx_g_extract.py <gpu> <tag> <seed> bank <i>/<n>   |   python code/sx_g_extract.py <gpu> <tag> <seed> sets
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dgcore, admwrap
from dgcore import ROOT, NPY, to_t, TF, shared_noise
from ct_extract_defs import SIGS, bank_indices
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
TIERS = {'lsb256': ('lsun_bedroom.pt', 'l256_bedroom_bank', 8), 'adm256c': ('256x256_diffusion_uncond.pt', 'j256_imagenet_train10', 8)}
dev = torch.device(f'cuda:{int(sys.argv[1])}'); tag = sys.argv[2]; seed = int(sys.argv[3]); what = sys.argv[4]; ck, bankset, bs = TIERS[tag]; sig = SIGS[0]; assert abs(sig - 0.01) < 1e-12
log = lambda *s: print(*s, flush=True); t0 = time.time()
FD = f'{ROOT}/runs/tc3/feat/s0/{tag}'; M = json.load(open(f'{FD}/meta.json')); BL, parts, ress, Cs, offs = M['blocks'], M['part'], M['res'], M['C'], M['off']; nB = len(BL)
A = np.array(json.load(open(f'{ROOT}/runs/tc3/depth_{tag}.json'))['A']); bu = int(A.argmax()); dec2 = [b for b in range(nB) if parts[b] == 'dec' and ress[b] == 2 * ress[bu]]; ba, bx = dec2[0], dec2[-1]; nb = bx + 1; C = Cs[bu]
OD = f'{ROOT}/runs/sx/g/feat/s{seed}/{tag}'; os.makedirs(OD, exist_ok=True); ORI = ['id', 'rot90', 'rot180', 'rot270', 'hflip', 'vflip', 'transp', 'atransp']
admwrap.CKPT = f'{ROOT}/ckpt/{ck}'; den, net = admwrap.load_adm(dev, fp16=True); mods = dict(net.named_modules()); store = {}
class Stop(Exception): pass
def mk(i):
    def f(m, inp, o):
        o = (o[0] if isinstance(o, (tuple, list)) else o).float()
        store[i] = (o - o.mean((2, 3), keepdim=True)).abs().amax((2, 3)).clamp_min(1e-20).log().mean(1)
        if i == bu: store['f'] = torch.cat([o.mean((2, 3)), o.std((2, 3), correction=0)], 1)
        if i == bx: raise Stop()
    return f
for i in range(nb): mods[BL[i]].register_forward_hook(mk(i))
@torch.no_grad()
def one(x):
    eps = shared_noise(x.shape[1:], seed, 0, 0, dev); store.clear()
    try: den(x + sig * eps, sig)
    except Stop: pass
    assert len(store) == nb + 1, len(store)
    return store['f'].half().cpu().numpy(), torch.stack([store[i] for i in range(nb)], 1).cpu().numpy()
meta = dict(tag=tag, seed=seed, ckpt=ck, bank=bankset, sigma=sig, n_blocks=nB, upright=BL[bu], i_upright=bu, res_upright=ress[bu], A=float(A[bu]), i_A=ba, exit=BL[bx], i_exit=bx, n_profile=nb, C=C, ori=ORI)
if what == 'bank':
    i0, n0 = [int(v) for v in sys.argv[5].split('/')]; full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); bidx, small = bank_indices(len(full)); ids = bidx[small]; N = len(ids); assert N == 10000
    sel = np.array_split(np.arange(N), n0)[i0]; Fo = np.empty((len(sel), 8, 2 * C), np.float16); Zo = np.empty((len(sel), 8, nb), np.float32)
    for i in range(0, len(sel), bs):
        x = to_t(np.ascontiguousarray(np.stack([full[j] for j in ids[sel[i:i + bs]]])), dev)
        for v, pre in enumerate(ORI): f, z = one(x if pre == 'id' else TF[pre][0](x)); Fo[i:i + len(x), v] = f; Zo[i:i + len(x), v] = z
        if i % (bs * 50) == 0: log('  bank', sys.argv[5], i, len(sel), f'{time.time() - t0:.0f}s')
    chk = float('nan')
    if seed == 0:                                    # the identity view against the stored full-pass features of PROTOCOL_TC3 (same seed)
        ref = np.load(f'{FD}/bank.npy', mmap_mode='r'); assert np.array_equal(np.load(f'{FD}/bank__idx.npy'), ids)
        k = min(200, len(sel)); chk = float(np.abs(np.asarray(ref[sel[:k], offs[bu]:offs[bu + 1]], np.float32) - Fo[:k, 0].astype(np.float32)).max())
    for nm, arr in (('f', Fo), ('z', Zo)): np.save(f'{OD}/bank_{nm}_p{i0}of{n0}.tmp.npy', arr); os.replace(f'{OD}/bank_{nm}_p{i0}of{n0}.tmp.npy', f'{OD}/bank_{nm}_p{i0}of{n0}.npy')
    json.dump(dict(meta, part=sys.argv[5], check_identity_vs_tc3_bank=chk), open(f'{OD}/meta_bank_p{i0}of{n0}.json', 'w'), indent=1)
    log('SX_G_EXTRACT_DONE', tag, seed, 'bank', sys.argv[5], Fo.shape, Zo.shape, f'U {bu + 1}, A {ba + 1}, X {bx + 1} of {nB}; identity view vs the TC3 bank (first {min(200, len(sel))}): max abs difference', chk, f'{time.time() - t0:.0f}s')
else:
    rows = [r for r in json.load(open(f'{ROOT}/runs/tc3/sets_{tag}.json')) if r[0].startswith('te-')]
    for name, data, idx, pre in rows:
        assert pre == 'none' or name.endswith('__' + pre), (name, pre); idx = np.asarray(idx, np.int64); xs = np.load(f'{NPY}/{data}.npy', mmap_mode='r'); Fo = np.empty((len(idx), 2 * C), np.float16); Zo = np.empty((len(idx), nb), np.float32)
        for i in range(0, len(idx), bs):
            x = to_t(np.ascontiguousarray(np.stack([xs[j] for j in idx[i:i + bs]])), dev); f, z = one(x if pre in ('none', 'id') else TF[pre][0](x)); Fo[i:i + len(x)] = f; Zo[i:i + len(x)] = z
        np.save(f'{OD}/{name}_f.npy', Fo); np.save(f'{OD}/{name}_z.npy', Zo); log('  set', name, len(idx), f'{time.time() - t0:.0f}s')
    json.dump(dict(meta, sets=[r[0] for r in rows]), open(f'{OD}/meta.json', 'w'), indent=1)
    log('SX_G_EXTRACT_DONE', tag, seed, 'sets', len(rows), f'{time.time() - t0:.0f}s')
