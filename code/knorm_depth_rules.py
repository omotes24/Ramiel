# knorm_depth_rules.py -- reads runs/knorm/depth/<tag>_s0.npz (knorm_depth.py).  Per block: intrinsic dimension d_b, how well the kNN term
#   of the block separates held-out upright train images from their seven transformed copies (label-free, no OOD image), the kNN term on
#   val1 and (ORACLE, development only) on test2.  Then rules that pick a block without OOD images, and sums over blocks.
#   usage: python code/knorm_depth_rules.py [tags]   -> runs/knorm/depth/rules_s0.json
import os, sys, json
import numpy as np
from scipy.stats import rankdata, spearmanr
ROOT = os.path.expanduser('~/ramiel'); OD = f'{ROOT}/runs/knorm/depth'
TAGS = sys.argv[1].split(',') if len(sys.argv) > 1 else ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']
IDN = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}


def auc(x, y): r = rankdata(np.concatenate([x, y])); return 100 * float((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y)))


OUT = {}
for t in TAGS:
    Z = np.load(f'{OD}/{t}_s0.npz'); M = json.load(open(f'{OD}/{t}_s0.json')); RS = M['RS']; nB = len(M['blocks']); nR = len(RS)
    lab = [f"b{b:02d}_{M['part'][b][0]}{M['res'][b]}" for b in range(nB)]; d = Z['d']
    idv = f'devall-{IDN[t]}'; sel = [n for n in M['val1'] if n.startswith('sel-')]; idt = f'test2-{IDN[t]}'
    nat = [n for n in M['test2'] if n != idt and '__' not in n and 'synth' not in n]
    L = {n: -np.log(Z[f'r__{n}'].astype(np.float64)) for n in [idv, idt, idt + '__vflip'] + sel + nat}          # [nB, N, nR]
    LV = -np.log(Z['rv'].astype(np.float64))                                                                 # [nB, 5000, 8, nR]
    v1 = np.array([[np.mean([auc(L[idv][b, :, j], L[n][b, :, j]) for n in sel]) for j in range(nR)] for b in range(nB)])
    t2 = np.array([[np.mean([auc(L[idt][b, :, j], L[n][b, :, j]) for n in nat]) for j in range(nR)] for b in range(nB)])
    vf = np.array([[auc(L[idt][b, :, j], L[idt + '__vflip'][b, :, j]) for j in range(nR)] for b in range(nB)])
    ori = np.array([[[auc(LV[b, :, 0, j], LV[b, :, o, j]) for o in range(1, 8)] for j in range(nR)] for b in range(nB)])   # [nB, nR, 7]
    om = ori.mean(2); jo = om.argmax(1); jv = v1.argmax(1)
    print(f'== {t}: per block  d | orientation AUROC (mean of 7, best R; rot90 / rot180 / vflip there) | val1 kNN (best R) | test2 at val1-R | test2 at ori-R | test2 oracle R')
    for b in range(nB):
        print(f"   {lab[b]:9s} C{M['C'][b]:<4d} d {d[b]:5.1f} | ori {om[b, jo[b]]:5.1f} R{RS[jo[b]]:<4d} ({ori[b, jo[b], 0]:.0f}/{ori[b, jo[b], 1]:.0f}/{ori[b, jo[b], 4]:.0f}) | "
              f"val1 {v1[b, jv[b]]:5.1f} R{RS[jv[b]]:<4d} | test2 {t2[b, jv[b]]:5.1f} | {t2[b, jo[b]]:5.1f} | {t2[b].max():5.1f} R{RS[t2[b].argmax()]}")
    dec = [b for b in range(nB) if M['part'][b] == 'dec']; allb = list(range(nB))
    res = {}
    def rule(name, b, j, note=''):
        res[name] = dict(block=lab[b], R=RS[j], val1=float(v1[b, j]), test2=float(t2[b, j]), vflip=float(vf[b, j]), d=float(d[b]))
        print(f"   rule {name:34s} -> {lab[b]} R{RS[j]:<4d} val1 {v1[b, j]:5.2f} | test2 {t2[b, j]:5.2f} (upside-down {vf[b, j]:5.1f}) {note}")
    bo = int(np.argmax(om.max(1))); rule('max orientation, R by orientation', bo, int(jo[bo])); rule('max orientation, R by val1', bo, int(jv[bo]))
    bod = dec[int(np.argmax(om[dec].max(1)))]; rule('max orientation (decoder), R ori', bod, int(jo[bod])); rule('max orientation (decoder), R val1', bod, int(jv[bod]))
    bd = int(np.argmin(d)); rule('min d, R by val1', bd, int(jv[bd])); bdd = dec[int(np.argmin(d[dec]))]; rule('min d (decoder), R by val1', bdd, int(jv[bdd]))
    for o, nm in ((0, 'rot90'), (1, 'rot180'), (4, 'vflip'), (3, 'hflip')):
        bb = int(np.argmax(ori[:, :, o].max(1))); rule(f'max {nm} AUROC, R by val1', bb, int(jv[bb]))
    bv = int(np.argmax(v1.max(1))); rule('max val1 (reference)', bv, int(jv[bv])); bt = int(np.argmax(t2.max(1))); rule('test2 oracle (reference)', bt, int(t2[bt].argmax()))
    sing = allb
    print(f"   rank correlation over blocks with test2 (best R): orientation {spearmanr(om.max(1), t2.max(1)).correlation:.2f} | val1 {spearmanr(v1.max(1), t2.max(1)).correlation:.2f} | -d {spearmanr(-d, t2.max(1)).correlation:.2f}"
          f" || with val1: orientation {spearmanr(om.max(1), v1.max(1)).correlation:.2f} | -d {spearmanr(-d, v1.max(1)).correlation:.2f}")
    # sums over blocks: sum_b -d_b log r_b (nats), shared R
    def sums(name, G, w=None):
        w = d[G] if w is None else w
        S = {n: np.tensordot(w, L[n][G], axes=(0, 0)) for n in [idv, idt, idt + '__vflip'] + sel + nat}            # [N, nR]
        a1 = np.array([np.mean([auc(S[idv][:, j], S[n][:, j]) for n in sel]) for j in range(nR)]); a2 = np.array([np.mean([auc(S[idt][:, j], S[n][:, j]) for n in nat]) for j in range(nR)])
        av = np.array([auc(S[idt][:, j], S[idt + '__vflip'][:, j]) for j in range(nR)]); j = int(a1.argmax())
        res['sum ' + name] = dict(R=RS[j], val1=float(a1[j]), test2=float(a2[j]), vflip=float(av[j]), test2_best=float(a2.max()), n=len(G))
        print(f"   sum  {name:34s} ({len(G):2d} blocks) R{RS[j]:<4d} val1 {a1[j]:5.2f} | test2 {a2[j]:5.2f} (upside-down {av[j]:5.1f}) | test2 over R: " + ' '.join(f'{x:.1f}' for x in a2))
    sums('all blocks', allb); sums('decoder', dec); sums('encoder', [b for b in allb if M['part'][b] == 'enc'])
    for r in sorted(set(M['res'])):
        G = [b for b in dec if M['res'][b] == r]
        if G: sums(f'decoder at {r}x{r}', G)
    k = max(3, nB // 5); top = list(np.argsort(-om.max(1))[:k]); sums(f'top {k} by orientation', sorted(int(b) for b in top))
    sums('all blocks, weight = orientation - 50', allb, w=d * np.clip(om.max(1) - 50, 0, None) / 50)
    OUT[t] = dict(labels=lab, d=d.tolist(), ori=om.tolist(), ori_all=ori.tolist(), val1=v1.tolist(), test2=t2.tolist(), vflip=vf.tolist(), rules=res, RS=RS)
json.dump(OUT, open(f'{OD}/rules_s0{"_" + "_".join(TAGS) if len(sys.argv) > 1 else ""}.json', 'w'))
print('RULES_DONE')
