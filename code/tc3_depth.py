# tc3_depth.py -- PROTOCOL_TC3.md: the upright-block rule on a new tier, and the test of its kNN term against the block that validation
#   OOD images choose.  Noise seed 0; [channel mean, channel std] of one block, unit length, cosine, single precision; the nested 10k
#   bank; R in {1, 2, 5, 10, 20, 50, 100, 200, 500, 1000}  (the form of knorm_depth.py / knorm_depth_rules.py).
#     A_{b,R}  = mean over the 7 non-identity symmetries g of AUROC(kNN term of the 5,000 held-out bank images vs. of their copies g.x),
#                the image's own bank entry excluded;  A_b = max_R A_{b,R};  upright block b* = argmax_b A_b   (ID training images only)
#     val1     : mean AUROC over the validation OOD sets (v1-ood-*) against v1-id, per block and R; R of a block = its best R on val1
#     test     : mean AUROC over the test OOD sets (te-ood-*) against te-id          (ONE run; nothing is chosen on test)
#   -> runs/tc3/depth_<tag>.json, runs/tc3/depth_<tag>.npz (the test kNN terms of the reported blocks)
#   usage: python code/tc3_depth.py <gpu> <tag>
import os, sys, json, time
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import rankdata, spearmanr
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
ROOT = os.path.expanduser('~/ramiel'); dev = torch.device(f'cuda:{int(sys.argv[1])}'); TAG = sys.argv[2]
fd = f'{ROOT}/runs/tc3/feat/s0/{TAG}'; RS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]; RI = [R - 1 for R in RS]; nR = len(RS)
M = json.load(open(f'{fd}/meta.json')); BL, Cs, parts, ress = M['blocks'], M['C'], M['part'], M['res']; nB = len(BL); coff = np.cumsum([0] + [2 * c for c in Cs])
log = lambda *s: print(*s, flush=True); t0 = time.time()
rows = json.load(open(f'{ROOT}/runs/tc3/sets_{TAG}.json')); names = [r[0] for r in rows]
v1o = [n for n in names if n.startswith('v1-ood-')]; teo = [n for n in names if n.startswith('te-ood-')]; assert 'v1-id' in names and 'te-id' in names and v1o and teo
BANK = np.load(f'{fd}/bank.npy', mmap_mode='r'); Z = {n: np.load(f'{fd}/{n}.npy', mmap_mode='r') for n in names}   # read block by block
for r in rows: assert np.array_equal(np.load(f'{fd}/{r[0]}__idx.npy'), np.asarray(r[2], np.int64)) and Z[r[0]].shape == (len(r[2]), coff[-1]), r[0]
vp = sorted(f for f in os.listdir(fd) if f.startswith('views_p') and f.endswith('.npy')); nv = int(vp[0].split('of')[1][:-4]); assert len(vp) == nv, vp
VWP = [np.load(f'{fd}/views_p{i}of{nv}.npy', mmap_mode='r') for i in range(nv)]; pos = np.load(f'{fd}/views__idx.npy'); assert sum(len(v) for v in VWP) == 5000 == len(pos) and all(v.shape[1:] == (8, coff[-1]) for v in VWP)
view = lambda v, sl: np.concatenate([np.asarray(q[:, v, sl]) for q in VWP])
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ct_extract_defs import bank_indices
bidx, small = bank_indices(len(np.load(f"{ROOT}/data/npy/{M['bank']}.npy", mmap_mode='r'))); bank_tr = np.load(f'{fd}/bank__idx.npy'); assert np.array_equal(bank_tr, bidx[small]) and len(BANK) == len(bank_tr)
where = {int(v): i for i, v in enumerate(bank_tr)}; own = torch.tensor([where.get(int(bidx[p]), -1) for p in pos], device=dev)
log(TAG, 'loaded: bank', BANK.shape, 'views', [v.shape for v in VWP], 'sets', len(names), '| held-out images in the bank:', int((own >= 0).sum()), f'{time.time() - t0:.0f}s')


def auc(x, y): r = rankdata(np.concatenate([x, y])); return 100 * float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


def knn(Q, B, own=None):
    out = []; k = min(max(RS), len(B) - (1 if own is not None else 0))
    for i in range(0, len(Q), 1024):
        g = Q[i:i + 1024] @ B.T
        if own is not None:
            o = own[i:i + 1024]; m = torch.nonzero(o >= 0).squeeze(1); g[m, o[m]] = -2.0
        out.append(torch.sqrt((2 * (1 - g.topk(k, dim=1).values)).clamp_min(2e-9))[:, RI])
    return torch.cat(out).cpu().numpy().astype(np.float64)


unit = lambda a: F.normalize(torch.from_numpy(np.ascontiguousarray(a)).to(dev).float(), dim=1)
ORI = np.zeros((nB, nR, 7)); V1 = np.zeros((nB, nR)); TE = np.zeros((nB, nR)); VF = np.zeros((nB, nR)); HF = np.zeros((nB, nR)); KEEP = {}
lab = [f'b{b:02d}_{parts[b][0]}{ress[b]}' for b in range(nB)]
for b in range(nB):
    sl = slice(coff[b], coff[b + 1]); B = unit(BANK[:, sl]); LV = np.stack([-np.log(knn(unit(view(v, sl)), B, own)) for v in range(8)], 1)   # [5000, 8, nR]
    L = {n: -np.log(knn(unit(Z[n][:, sl]), B)) for n in names}; KEEP[b] = {n: L[n].astype(np.float32) for n in names if n.startswith('te-')}
    for j in range(nR):
        ORI[b, j] = [auc(LV[:, 0, j], LV[:, o, j]) for o in range(1, 8)]
        V1[b, j] = np.mean([auc(L['v1-id'][:, j], L[n][:, j]) for n in v1o]); TE[b, j] = np.mean([auc(L['te-id'][:, j], L[n][:, j]) for n in teo])
        VF[b, j] = auc(L['te-id'][:, j], L['te-id__vflip'][:, j]); HF[b, j] = auc(L['te-id'][:, j], L['te-id__hflip'][:, j]) if 'te-id__hflip' in L else np.nan
    log(TAG, f'{lab[b]:9s} C{Cs[b]:<4d} done', f'{time.time() - t0:.0f}s')
om = ORI.mean(2); A = om.max(1); jo = om.argmax(1); jv = V1.argmax(1); up = int(A.argmax()); pv1 = V1[np.arange(nB), jv]; pte = TE[np.arange(nB), jv]; bv = int(pv1.argmax())
Rin = max(ress); last = M['last'] if 'last' in M else [[b for b in range(nB) if parts[b] == 'dec' and ress[b] == Rin >> l][-1] for l in range(3)]   # a transformer has no decoder resolutions: its meta names L0, L1, L2
# the last-block reference of the resume: the mean cosine of the three last decoder blocks (L0 + L1 + L2), R by val1
B3 = torch.cat([unit(BANK[:, coff[b]:coff[b + 1]]) for b in last], 1) / np.sqrt(3)
L3 = {n: -np.log(knn(torch.cat([unit(Z[n][:, coff[b]:coff[b + 1]]) for b in last], 1) / np.sqrt(3), B3)) for n in names}
v3 = [np.mean([auc(L3['v1-id'][:, j], L3[n][:, j]) for n in v1o]) for j in range(nR)]; j3 = int(np.argmax(v3)); te3 = float(np.mean([auc(L3['te-id'][:, j3], L3[n][:, j3]) for n in teo]))
for b in range(nB):
    log(f"   {lab[b]:9s} A {A[b]:5.1f} (R{RS[jo[b]]:<4d}; rot90/rot180/hflip/vflip {ORI[b, jo[b], 0]:.0f}/{ORI[b, jo[b], 1]:.0f}/{ORI[b, jo[b], 3]:.0f}/{ORI[b, jo[b], 4]:.0f}) | val1 {pv1[b]:5.1f} R{RS[jv[b]]:<4d} | "
        f"test {pte[b]:5.1f} | test at the R of A {TE[b, jo[b]]:5.1f} | upside-down {VF[b, jv[b]]:5.1f}" + ('   <- upright' if b == up else '') + ('   <- val1' if b == bv else ''))
# paired bootstrap of the difference (upright block - val1-chosen block) of the test mean AUROC: ID and OOD images resampled, the same images for both
rng = np.random.RandomState(20261123); KU, KV = KEEP[up], KEEP[bv]; ju, jb = int(jv[up]), int(jv[bv]); nid = len(KU['te-id']); dif = []
for _ in range(2000):
    ii = rng.randint(0, nid, nid); dd = []
    for n in teo:
        jj = rng.randint(0, len(KU[n]), len(KU[n])); dd.append(auc(KU['te-id'][ii, ju], KU[n][jj, ju]) - auc(KV['te-id'][ii, jb], KV[n][jj, jb]))
    dif.append(np.mean(dd))
lo, hi = np.quantile(dif, [0.025, 0.975]); per = lambda b, j: {n[len('te-ood-'):]: auc(KEEP[b]['te-id'][:, j], KEEP[b][n][:, j]) for n in teo}
out = dict(tag=TAG, labels=lab, blocks=BL, part=parts, res=ress, C=Cs, RS=RS, A=A.tolist(), A_R=[int(RS[j]) for j in jo], A_by_g=[ORI[b, jo[b]].tolist() for b in range(nB)],
           val1=pv1.tolist(), val1_R=[int(RS[j]) for j in jv], test=pte.tolist(), test_all=TE.tolist(), val1_all=V1.tolist(), vflip=VF[np.arange(nB), jv].tolist(), hflip=HF[np.arange(nB), jv].tolist(),
           upright=up, val1_block=bv, last=last, v1_sets=v1o, te_sets=teo,
           test_upright=float(pte[up]), test_val1_block=float(pte[bv]), diff=float(pte[up] - pte[bv]), diff_ci=[float(lo), float(hi)], test_upright_R_by_A=float(TE[up, jo[up]]),
           test_best=float(pte.max()), test_best_block=int(pte.argmax()), test_last=[float(pte[b]) for b in last], test_L012=te3, L012_R=int(RS[j3]), val1_L012=float(v3[j3]),
           per_set_upright=per(up, ju), per_set_val1_block=per(bv, jb), vflip_upright=float(VF[up, ju]), vflip_val1_block=float(VF[bv, jb]), hflip_upright=float(HF[up, ju]),
           rho_A_test=float(spearmanr(A, pte).correlation), rho_val1_test=float(spearmanr(pv1, pte).correlation), rho_A_val1=float(spearmanr(A, pv1).correlation))
json.dump(out, open(f'{ROOT}/runs/tc3/depth_{TAG}.json', 'w'))
np.savez(f'{ROOT}/runs/tc3/depth_{TAG}.npz', **{f'{w}|{n}': KEEP[b][n][:, j] for w, b, j in (('upright', up, ju), ('val1', bv, jb)) for n in KEEP[b]})
log(f"== {TAG}: upright block {lab[up]} (A {A[up]:.1f}, R {RS[ju]}) test {pte[up]:.2f} | val1-chosen block {lab[bv]} (val1 {pv1[bv]:.1f}, R {RS[jb]}) test {pte[bv]:.2f} | difference {pte[up] - pte[bv]:+.2f} [{lo:+.2f}, {hi:+.2f}]")
log(f"   upright block with the R of A: {TE[up, jo[up]]:.2f} | best block on test {lab[int(pte.argmax())]} {pte.max():.2f} | last decoder blocks {[round(float(pte[b]), 1) for b in last]} | L0+L1+L2 {te3:.2f} (R {RS[j3]})")
log(f"   rho(A, test) {out['rho_A_test']:.2f} | rho(val1, test) {out['rho_val1_test']:.2f} | upside-down: upright {VF[up, ju]:.1f}, val1-chosen {VF[bv, jb]:.1f}")
log('TC3_DEPTH_DONE', TAG, f'{time.time() - t0:.0f}s')
