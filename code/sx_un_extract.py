# sx_un_extract.py -- the bank in eight orientations for the detector WITHOUT training: [channel mean, channel std] of the upright block
#   for every bank image and each of its eight orientations (id, rot90, rot180, rot270, hflip, vflip, transp, atransp), from one pass
#   stopped at the upright block (noise seed given; the image is transformed, the shared noise image is not).
#   output: runs/sx/nt/feat/s<seed>/<tag>/union.npy [N, 8, 2C] float16.   usage: python code/sx_un_extract.py <gpu> <tag> <seed>
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dgcore
from dgcore import ROOT, NPY, to_t, TF, shared_noise
from ct_extract_defs import SPEC, SIGS, bank_indices
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = torch.device(f'cuda:{int(sys.argv[1])}'); tag = sys.argv[2]; seed = int(sys.argv[3]); spec, bankset, bs = SPEC[tag]; sig = SIGS[0]; t0 = time.time()
UP = f'{ROOT}/runs/knorm/up/s{seed}/{tag}'; MU_ = json.load(open(f'{UP}/meta.json')); BN, C = MU_['name'], MU_['C']; OD = f'{ROOT}/runs/sx/nt/feat/s{seed}/{tag}'; os.makedirs(OD, exist_ok=True)
ORI = ['id', 'rot90', 'rot180', 'rot270', 'hflip', 'vflip', 'transp', 'atransp']; den, net = dgcore.load_model(spec, dev); mods = dict(net.named_modules()); store = {}
class Stop(Exception): pass
def grab(m, inp, o):
    store['o'] = o[0] if isinstance(o, (tuple, list)) else o; raise Stop()
mods[BN].register_forward_hook(grab)
@torch.no_grad()
def feats(x):
    eps = shared_noise(x.shape[1:], seed, 0, 0, dev); store.clear()
    try: den(x + sig * eps, sig)
    except Stop: pass
    o = store['o'].float(); return torch.cat([o.mean((2, 3)), o.std((2, 3), correction=0)], 1).half().cpu().numpy()
full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); bidx, small = bank_indices(len(full)); N = len(bidx); out = np.empty((N, 8, 2 * C), np.float16)
for i in range(0, N, bs):
    x = to_t(np.stack([full[j] for j in bidx[i:i + bs]]), dev)
    for v, pre in enumerate(ORI): out[i:i + len(x), v] = feats(x if pre == 'id' else TF[pre][0](x))
ref = np.load(f'{UP}/bank.npy', mmap_mode='r'); chk = float(np.abs(np.asarray(ref[:2000], np.float32) - out[:2000, 0].astype(np.float32)).max())
np.save(f'{OD}/union.tmp.npy', out); os.replace(f'{OD}/union.tmp.npy', f'{OD}/union.npy'); json.dump(dict(tag=tag, seed=seed, block=BN, C=C, N=N, ori=ORI, check_identity_vs_stored_bank=chk), open(f'{OD}/meta.json', 'w'))
print('SX_UN_EXTRACT_DONE', tag, seed, BN, out.shape, 'identity view vs the stored bank (first 2,000): max abs difference', chk, f'{time.time() - t0:.0f}s', flush=True)
