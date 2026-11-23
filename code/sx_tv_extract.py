# sx_tv_extract.py -- for the detector WITHOUT training with two passes: [channel mean, channel std] of the upright block for every
#   test2 / testc image as it is given (view 0) and turned by 180 degrees (view 1); each pass stops at the upright block (noise seed given).
#   output: runs/sx/nt/feat/s<seed>/<tag>/tv/<set>.npy [n, 2, 2C] float16.   usage: python code/sx_tv_extract.py <gpu> <tag> <seed>
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dgcore
from dgcore import ROOT, NPY, to_t, TF, shared_noise
from ct_extract_defs import SPEC, SIGS
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = torch.device(f'cuda:{int(sys.argv[1])}'); tag = sys.argv[2]; seed = int(sys.argv[3]); spec, bankset, bs = SPEC[tag]; sig = SIGS[0]; t0 = time.time()
IDN = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
UP = f'{ROOT}/runs/knorm/up/s{seed}/{tag}'; MU_ = json.load(open(f'{UP}/meta.json')); BN, C = MU_['name'], MU_['C']; OD = f'{ROOT}/runs/sx/nt/feat/s{seed}/{tag}/tv'; os.makedirs(OD, exist_ok=True)
den, net = dgcore.load_model(spec, dev); mods = dict(net.named_modules()); store = {}
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
rows = json.load(open(f'{ROOT}/runs/e11/sets_{tag}.json')); orc = json.load(open(f'{ROOT}/runs/orbc/sets_{tag}.json')); tc = [r for r in orc if r[0].startswith('testc-')]; tid = [r for r in tc if r[0] == f'testc-{IDN[tag]}']; assert len(tid) == 1
want = [r for r in rows if r[0].startswith('test2-')] + tc + [[tid[0][0] + '__vflip', tid[0][1], tid[0][2], 'vflip'], [tid[0][0] + '__hflip', tid[0][1], tid[0][2], 'hflip']]; worst = 0.0
for name, data, idx, pre in want:
    assert pre in ('none', 'id') or name.endswith('__' + pre), (name, pre); xs = np.load(f'{NPY}/{data}.npy', mmap_mode='r'); idx = np.asarray(idx, np.int64); out = np.empty((len(idx), 2, 2 * C), np.float16)
    for i in range(0, len(idx), bs):
        x = to_t(np.stack([xs[j] for j in idx[i:i + bs]]), dev); x = x if pre in ('none', 'id') else TF[pre][0](x); out[i:i + len(x), 0] = feats(x); out[i:i + len(x), 1] = feats(TF['rot180'][0](x))
    ref = np.load(f'{UP}/{name}.npy'); assert ref.shape == out[:, 0].shape, name; worst = max(worst, float(np.abs(ref.astype(np.float32) - out[:, 0].astype(np.float32)).max())); np.save(f'{OD}/{name}.npy', out)
print('SX_TV_EXTRACT_DONE', tag, seed, BN, len(want), 'sets; view 0 against the stored upright-block features: max abs difference', worst, f'{time.time() - t0:.0f}s', flush=True)
