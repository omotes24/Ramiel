# sx_e_bench.py -- time per image of the detector of addendum E on one GPU (batch 128; median of 8 batches after 3 warm-up batches;
#   torch.cuda.synchronize): the network up to U / A / X against the full pass, and the nearest-neighbour searches in the banks
#   (f: N upright entries, ranks to 1000; 7N re-oriented entries, ranks to 7000; z: the same two banks in the profile space).
#   usage: python code/sx_e_bench.py <gpu> <tag>     -> runs/sx/e_paper/bench_<tag>.json
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dgcore import ROOT, NPY, load_model, to_t, shared_noise
from ct_extract_defs import SPEC
dev = torch.device(f'cuda:{int(sys.argv[1])}'); tag = sys.argv[2]; SIG = 0.01; BS = 128; NW, NT = 3, 8; OD = f'{ROOT}/runs/sx/e_paper'; os.makedirs(OD, exist_ok=True)
spec, bankset, _ = SPEC[tag]; den, net = load_model(spec, dev); mods = dict(net.named_modules()); FD = f'{ROOT}/runs/sx/nt/feat/s0/{tag}'
M = json.load(open(f'{ROOT}/runs/up/feat/s0/{tag}/meta.json')); MA = json.load(open(f'{FD}/ap/meta.json')); BL = M['blocks']; iu, ia, ix = MA['i_upright'], MA['i_A'], MA['i_exit']; STOP = [-1]; store = {}
class Stop(Exception): pass
def mk(i):
    def hook(m, inp, o):
        o = (o[0] if isinstance(o, (tuple, list)) else o).float()
        if STOP[0] >= 0 and i <= STOP[0]:                 # the two descriptors are computed inside the timed pass
            store[i] = (o - o.mean((2, 3), keepdim=True)).abs().amax((2, 3)).clamp_min(1e-20).log().mean(1)
            if i == iu: store['f'] = F.normalize(torch.cat([o.mean((2, 3)), o.std((2, 3), correction=0)], 1), dim=1)
        if STOP[0] == i: raise Stop()
    return hook
for i, n in enumerate(BL): mods[n].register_forward_hook(mk(i))
full = np.load(f'{NPY}/{bankset}.npy', mmap_mode='r'); xs = [to_t(np.ascontiguousarray(full[i * BS:(i + 1) * BS]), dev) for i in range(NW + NT)]; eps = shared_noise(xs[0].shape[1:], 0, 0, 0, dev)
def timed(fn):
    ts = []
    for j in range(NW + NT):
        torch.cuda.synchronize(dev); t = time.perf_counter(); fn(j); torch.cuda.synchronize(dev)
        if j >= NW: ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts)) / BS            # ms per image
def net_to(stop):
    def fn(j):
        STOP[0] = stop; store.clear()
        try: den(xs[j] + SIG * eps, SIG)
        except Stop: pass
    return fn
RS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]; RO = [7 * r for r in RS]
UNI = np.load(f'{FD}/union.npy'); N, _, D_ = UNI.shape; BU = F.normalize(torch.from_numpy(np.ascontiguousarray(UNI.transpose(1, 0, 2).reshape(8 * N, D_))).to(dev).float(), dim=-1); del UNI
AP = np.load(f'{FD}/ap/bank.npy'); mu_p, sd_p = AP[:, 0].mean(0), AP[:, 0].std(0) + 1e-6; E = torch.from_numpy(((AP - mu_p) / sd_p).transpose(1, 0, 2).reshape(8 * N, -1)).to(dev).float(); n2 = (E * E).sum(1)
qf = BU[torch.randperm(N, device=dev)[:BS * (NW + NT)]].reshape(NW + NT, BS, -1); qz = E[torch.randperm(N, device=dev)[:BS * (NW + NT)]].reshape(NW + NT, BS, -1)
def s_f(lo, hi, ranks):
    def fn(j): d2 = 2 - 2 * qf[j] @ BU[lo:hi].T; (-(-d2).topk(max(ranks), dim=1).values[:, [r - 1 for r in ranks]]).clamp_min(1e-12).log()
    return fn
def s_z(lo, hi, ranks):
    def fn(j): q = qz[j]; d2 = (q * q).sum(1, keepdim=True) + n2[None, lo:hi] - 2 * q @ E[lo:hi].T; (-(-d2).topk(max(ranks), dim=1).values[:, [r - 1 for r in ranks]]).clamp_min(1e-12).log()
    return fn
with torch.no_grad():
    timed(net_to(-1)); R = dict(tag=tag, N=N, dim_f=D_, dim_z=int(E.shape[1]), batch=BS, i_upright=iu, i_A=ia, i_exit=ix,
                                full=timed(net_to(-1)), to_U=timed(net_to(iu)), to_A=timed(net_to(ia)), to_X=timed(net_to(ix)),
                                f_u=timed(s_f(0, N, RS)), f_o=timed(s_f(N, 8 * N, RO)), z_u=timed(s_z(0, N, RS)), z_o=timed(s_z(N, 8 * N, RO)))
    R['full2'] = timed(net_to(-1))
R['ours_X'] = R['to_X'] + R['f_u'] + R['f_o'] + R['z_u'] + R['z_o']; R['ours_U'] = R['to_U'] + R['f_u'] + R['f_o'] + R['z_u'] + R['z_o']; R['ours_A'] = R['to_A'] + R['f_u'] + R['f_o'] + R['z_u'] + R['z_o']; R['knn_full'] = R['full'] + R['f_u']
json.dump(R, open(f'{OD}/bench_{tag}.json', 'w'), indent=1)
print(tag, ' '.join(f'{k} {v:.3f}' for k, v in R.items() if isinstance(v, float)), flush=True)
