# knorm_up_extract.py -- [channel mean, channel std] of ONE block from one pass at sigma_1 (x + sigma_1 eps, eps = the shared noise image of
#   the seed; the image is transformed, the noise is not) for PROTOCOL_UPBLOCK.md:
#   the FULL bank (the reference images of the kNN term: bank_indices, 50k; ImageNet-64 10k), the val1 ID images (devall-<ID>), the val1
#   OOD sets (sel-; seed 0 only, they choose R), every test2 set (runs/e11/sets_<tag>.json, pre applied), every testc set
#   (runs/orbc/sets_<tag>.json) and the upside-down and mirrored testc ID images.
#   output: runs/knorm/up/s<seed>/<tag>/<name>.npy [N, 2C] float16, <name>__idx.npy, meta.json
#   usage: python code/knorm_up_extract.py --tag edmc10 --block 20 --gpu 0 --seed 0
import os, sys, json, time, argparse
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dgcore
from dgcore import ROOT, NPY, to_t, TF, shared_noise
from ct_extract_defs import SPEC, SIGS, bank_indices
p = argparse.ArgumentParser(); p.add_argument('--tag', required=True); p.add_argument('--block', type=int, required=True)
p.add_argument('--gpu', type=int, default=0); p.add_argument('--seed', type=int, default=0)
a = p.parse_args(); spec, bankset, bs = SPEC[a.tag]; k = 0; sig = SIGS[k]
IDN = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
dev = torch.device(f'cuda:{a.gpu}'); torch.backends.cudnn.benchmark = True
M = json.load(open(f'{ROOT}/runs/up/feat/s0/{a.tag}/meta.json')); BN = M['blocks'][a.block]; C = M['C'][a.block]; OFF = M['off'][a.block]
den, net = dgcore.load_model(spec, dev); mods = dict(net.named_modules()); store = {}


def grab(m, inp, o):
    if isinstance(o, (tuple, list)): o = o[0]
    store['o'] = o


mods[BN].register_forward_hook(grab)


@torch.no_grad()
def feats(x):
    eps = shared_noise(x.shape[1:], a.seed, k, 0, dev); store.clear(); den(x + sig * eps, sig); o = store['o'].float()
    return torch.cat([o.mean((2, 3)), o.std((2, 3), correction=0)], 1).half().cpu().numpy()


od = f'{ROOT}/runs/knorm/up/s{a.seed}/{a.tag}'; os.makedirs(od, exist_ok=True); t0 = time.time()
json.dump(dict(tag=a.tag, seed=a.seed, block=a.block, name=BN, C=C, part=M['part'][a.block], res=M['res'][a.block], sigma=sig), open(f'{od}/meta.json', 'w'))


def run(name, xs, idx, pre):
    fn = f'{od}/{name}.npy'
    if os.path.exists(fn): print('exists', fn, flush=True); return
    idx = np.asarray(idx, np.int64); out = np.empty((len(idx), 2 * C), np.float16)
    for i in range(0, len(idx), bs):
        x = to_t(np.stack([xs[j] for j in idx[i:i + bs]]), dev); out[i:i + len(x)] = feats(x if pre in ('none', 'id') else TF[pre][0](x))
    np.save(f'{od}/{name}__idx.npy', idx); np.save(f'{od}/{name}.tmp.npy', out); os.replace(f'{od}/{name}.tmp.npy', fn)
    print(a.tag, a.seed, BN, name, len(idx), pre, f'{time.time() - t0:.0f}s', flush=True)


full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); bidx, small = bank_indices(len(full)); run('bank', full, bidx, 'none')
rows = json.load(open(f'{ROOT}/runs/ok/sets_{a.tag}.json')); e11 = json.load(open(f'{ROOT}/runs/e11/sets_{a.tag}.json')); orc = json.load(open(f'{ROOT}/runs/orbc/sets_{a.tag}.json'))
tc = [r for r in orc if r[0].startswith('testc-')]; tid = [r for r in tc if r[0] == f'testc-{IDN[a.tag]}']; assert len(tid) == 1
want = [r for r in rows if r[0] == f'devall-{IDN[a.tag]}'] + ([r for r in rows if r[0].startswith('sel-')] if a.seed == 0 else []) + \
       [r for r in e11 if r[0].startswith('test2-')] + tc + [[tid[0][0] + '__vflip', tid[0][1], tid[0][2], 'vflip'], [tid[0][0] + '__hflip', tid[0][1], tid[0][2], 'hflip']]
for name, data, idx, pre in want:
    assert pre in ('none', 'id') or name.endswith('__' + pre), (name, pre)
    run(name, np.load(f'{NPY}/{data}.npy', mmap_mode='r'), idx, pre)
if a.seed == 0:                                                     # the same numbers as runs/up/feat (the screen's features)?
    X = np.load(f'{ROOT}/runs/up/feat/s0/{a.tag}/devall-{IDN[a.tag]}.npy', mmap_mode='r'); Z = np.load(f'{od}/devall-{IDN[a.tag]}.npy').astype(np.float32)
    print('check vs runs/up/feat', a.tag, 'devall max abs', float(np.abs(np.asarray(X[:, OFF:OFF + 2 * C], np.float32) - Z).max()), flush=True)
    bi = np.load(f'{ROOT}/runs/up/feat/s0/{a.tag}/bank__idx.npy'); pos = np.searchsorted(bidx, bi); assert np.array_equal(bidx[pos], bi)
    Xb = np.load(f'{ROOT}/runs/up/feat/s0/{a.tag}/bank.npy', mmap_mode='r'); Zb = np.load(f'{od}/bank.npy', mmap_mode='r')
    print('check vs runs/up/feat', a.tag, 'bank (first 300 of the nested bank) max abs', float(np.abs(np.asarray(Xb[:300, OFF:OFF + 2 * C], np.float32) - np.asarray(Zb[pos[:300]], np.float32)).max()), flush=True)
print('KNORM_UP_EXTRACT_DONE', a.tag, a.seed, BN, f'{time.time() - t0:.0f}s', flush=True)
