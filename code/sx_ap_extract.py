# sx_ap_extract.py -- amplitude profiles for the detector with ONE forward pass and NO training.
#   a_h(x) = channel mean of log max_u |f_h - mean_u f_h| for every block h of the pass up to the exit block X (last decoder block at twice
#   the resolution of the upright block); the pass stops at X; sigma_1 = 0.01, shared noise image of the seed (the image is transformed,
#   the noise is not).  Extracted for: every bank image in its eight orientations (id, rot90, rot180, rot270, hflip, vflip, transp,
#   atransp), the val1 / val2 sets (seed 0 only; devall-, sel-, fresh-), every test2 / testc set as it is given.
#   output: runs/sx/nt/feat/s<seed>/<tag>/ap/{bank.npy [N, 8, nb], <set>.npy [n, nb], meta.json}.   usage: python code/sx_ap_extract.py <gpu> <tag> <seed>
import os, sys, json, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dgcore
from dgcore import ROOT, NPY, to_t, TF, shared_noise
from ct_extract_defs import SPEC, SIGS, bank_indices
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
dev = torch.device(f"cuda:{int(sys.argv[1])}"); tag = sys.argv[2]; seed = int(sys.argv[3]); spec, bankset, bs = SPEC[tag]; sig = SIGS[0]; t0 = time.time(); log = lambda *s: print(*s, flush=True)
IDN = {"edmc10": "cifar10_test", "l6svhn": "svhn_test", "dpceleba": "celeba_test", "in64": "in64_imagenet_test", "fmnist": "g32_fmnist_test"}
MU_ = json.load(open(f"{ROOT}/runs/knorm/up/s{seed}/{tag}/meta.json")); M = json.load(open(f"{ROOT}/runs/up/feat/s0/{tag}/meta.json")); BL, parts, ress, Cs, offs = M["blocks"], M["part"], M["res"], M["C"], M["off"]; nB = len(BL)
bu = BL.index(MU_["name"]); bx = [b for b in range(nB) if parts[b] == "dec" and ress[b] == 2 * ress[bu]][-1]; ba = [b for b in range(nB) if parts[b] == "dec" and ress[b] == 2 * ress[bu]][0]; nb = bx + 1
OD = f"{ROOT}/runs/sx/nt/feat/s{seed}/{tag}/ap"; os.makedirs(OD, exist_ok=True); ORI = ["id", "rot90", "rot180", "rot270", "hflip", "vflip", "transp", "atransp"]
den, net = dgcore.load_model(spec, dev); mods = dict(net.named_modules()); store = {}
class Stop(Exception): pass
def mk(i):
    def f(m, inp, o):
        o = (o[0] if isinstance(o, (tuple, list)) else o).float(); store[i] = (o - o.mean((2, 3), keepdim=True)).abs().amax((2, 3)).clamp_min(1e-20).log().mean(1)
        if i == bx: raise Stop()
    return f
for i in range(nb): mods[BL[i]].register_forward_hook(mk(i))
@torch.no_grad()
def prof(x):
    eps = shared_noise(x.shape[1:], seed, 0, 0, dev); store.clear()
    try: den(x + sig * eps, sig)
    except Stop: pass
    return torch.stack([store[i] for i in range(nb)], 1).cpu().numpy()
def run(xs, idx, pre):
    idx = np.asarray(idx, np.int64); out = np.empty((len(idx), nb), np.float32)
    for i in range(0, len(idx), bs):
        x = to_t(np.stack([xs[j] for j in idx[i:i + bs]]), dev); out[i:i + len(x)] = prof(x if pre in ("none", "id") else TF[pre][0](x))
    return out
full = np.load(f"{NPY}/{bankset}.npy", mmap_mode="r"); bidx, small = bank_indices(len(full)); N = len(bidx); chk = float("nan")
if not os.path.exists(f"{OD}/bank.npy"):
    out = np.empty((N, 8, nb), np.float32)
    for i in range(0, N, bs):
        x = to_t(np.stack([full[j] for j in bidx[i:i + bs]]), dev)
        for v, o in enumerate(ORI): out[i:i + len(x), v] = prof(x if o == "id" else TF[o][0](x))
    np.save(f"{OD}/bank.tmp.npy", out); os.replace(f"{OD}/bank.tmp.npy", f"{OD}/bank.npy"); log(tag, seed, "bank x 8", out.shape, f"{time.time() - t0:.0f}s")
    if seed == 0:                                     # the cached per-channel log-amplitudes of the screen give the same profile?
        fd = f"{ROOT}/runs/up/feat/s0/{tag}"; pos = np.load(f"{fd}/views__idx.npy")[:100]; ref = np.load(f"{fd}/views.npy", mmap_mode="r")
        r = np.stack([np.asarray(ref[:100, :, offs[b] + 2 * Cs[b]:offs[b] + 3 * Cs[b]], np.float32).mean(-1) for b in range(nb)], -1); chk = float(np.abs(r - out[pos]).max()); log(tag, "check against the cached log-amplitudes (100 images x 8): max abs difference", chk)
rows = []
if seed == 0: rows += [r for r in json.load(open(f"{ROOT}/runs/ok/sets_{tag}.json")) if r[0].startswith(("devall-", "sel-", "fresh-")) and os.path.exists(f"{ROOT}/runs/up/feat/s0/{tag}/{r[0]}__idx.npy")]
e11 = json.load(open(f"{ROOT}/runs/e11/sets_{tag}.json")); orc = json.load(open(f"{ROOT}/runs/orbc/sets_{tag}.json")); tc = [r for r in orc if r[0].startswith("testc-")]; tid = [r for r in tc if r[0] == f"testc-{IDN[tag]}"]; assert len(tid) == 1
rows += [r for r in e11 if r[0].startswith(("test2-", "fresh-")) and r[3] not in ("none", "id") and (seed == 0 or r[0].startswith("test2-"))] + [r for r in e11 if r[0].startswith("test2-") and r[3] in ("none", "id")] + tc + [[tid[0][0] + "__vflip", tid[0][1], tid[0][2], "vflip"], [tid[0][0] + "__hflip", tid[0][1], tid[0][2], "hflip"]]
done = 0
for name, data, idx, pre in rows:
    if os.path.exists(f"{OD}/{name}.npy") or not os.path.exists(f"{NPY}/{data}.npy"): continue
    np.save(f"{OD}/{name}.npy", run(np.load(f"{NPY}/{data}.npy", mmap_mode="r"), idx, pre)); done += 1
json.dump(dict(tag=tag, seed=seed, N=N, n_profile=nb, upright=BL[bu], i_upright=bu, i_A=ba, exit=BL[bx], i_exit=bx, n_blocks=nB, ori=ORI, check=chk), open(f"{OD}/meta.json", "w"))
log("SX_AP_EXTRACT_DONE", tag, seed, f"profile of {nb} blocks (U {bu + 1}, A {ba + 1}, X {bx + 1} of {nB});", done, "sets", f"{time.time() - t0:.0f}s")
