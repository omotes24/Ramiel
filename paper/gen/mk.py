# mk.py -- numbers, table bodies and the two data figures of the paper, written from data/*.json (copied byte for byte from the GPU
#   server; md5 checked).  Server names of the image splits: test2 = test, testc = test'.
#   writes ../numbers.tex, ../tab_main.tex, ../tab_parts.tex, ../tab_cfs.tex, ../supp_*.tex, ../figs/fig_profile.tex, ../figs/fig_scatter.tex
import json, os
from decimal import Decimal, ROUND_HALF_UP
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.dirname(HERE); D = os.path.join(OUT, 'data')
S = json.load(open(f'{D}/sum_main.json')); FIX = json.load(open(f'{D}/sum_fix.json')); P = {'test2': json.load(open(f'{D}/sum_pairs2.json')), 'testc': json.load(open(f'{D}/sum_pairsc.json'))}
FG = json.load(open(f'{D}/fig_small.json')); PASS = json.load(open(f'{D}/sum_pass.json')); E1 = json.load(open(f'{D}/e1b_an.json'))
TAGS = ['edmc10', 'l6svhn', 'dpceleba', 'in64', 'fmnist']
NAME = {'edmc10': 'CIFAR-10', 'l6svhn': 'SVHN', 'dpceleba': 'CelebA', 'in64': 'ImageNet-64', 'fmnist': 'FashionMNIST'}
IDD = {'edmc10': 'cifar10_test', 'l6svhn': 'svhn_test', 'dpceleba': 'celeba_test', 'in64': 'in64_imagenet_test', 'fmnist': 'g32_fmnist_test'}
NAT = {'edmc10': ['svhn_test', 'textures', 'celeba_test', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val', 'cifar100_test'],
       'l6svhn': ['cifar10_test', 'cifar100_test', 'celeba_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val'],
       'dpceleba': ['cifar10_test', 'cifar100_test', 'svhn_test', 'textures', 'tin_val', 'lsun_fix', 'imagenet_fix', 'imagenet32_val', 'places365_val'],
       'in64': ['in64_ssb_hard', 'in64_ninco', 'in64_inaturalist', 'in64_textures', 'in64_openimage_o', 'in64_places365', 'in64_celeba_test'],
       'fmnist': ['g32_mnist_test', 'g32_kmnist_test', 'g32_omniglot_eval', 'g32_notmnist']}
SETN = {'svhn_test': 'SVHN', 'textures': 'Textures', 'celeba_test': 'CelebA', 'tin_val': 'Tiny-ImageNet', 'lsun_fix': 'LSUN', 'imagenet_fix': 'ImageNet',
        'imagenet32_val': 'ImageNet32', 'places365_val': 'Places365', 'cifar100_test': 'CIFAR-100', 'cifar10_test': 'CIFAR-10',
        'in64_ssb_hard': 'SSB-hard', 'in64_ninco': 'NINCO', 'in64_inaturalist': 'iNaturalist', 'in64_textures': 'Textures', 'in64_openimage_o': 'OpenImage-O',
        'in64_places365': 'Places365', 'in64_celeba_test': 'CelebA', 'g32_mnist_test': 'MNIST', 'g32_kmnist_test': 'KMNIST', 'g32_omniglot_eval': 'Omniglot',
        'g32_notmnist': 'notMNIST', 'synth_noise': '一様ノイズ', 'synth_const': '定数画像', 'g32_synth_noise': '一様ノイズ', 'g32_synth_const': '定数画像'}
PRIOR = ['NLL', 'IC-png', 'IC-j2k', 'TT', 'MSMA*', 'DiffPath', 'GEPC', 'EigenScore', 'CFS', 'CFS-kNN']
LABEL = {'NLL': 'NLL(尤度)', 'IC-png': 'IC(PNG)', 'IC-j2k': 'IC(JPEG2000)', 'TT': 'TT', 'MSMA*': 'MSMA型', 'DiffPath': 'DiffPath', 'GEPC': 'GEPC', 'EigenScore': 'EigenScore',
         'RotCls4': '参考：回転の予測(ResNet-18)', 'kNN (CT)': 'kNN~\\cite{sun2022knn}', 'CFS': 'CFS', 'CFS-kNN': 'CFS(kNN ヘッド)'}
SHORT = {'NLL': 'NLL', 'IC-png': 'IC-PNG', 'IC-j2k': 'IC-J2K', 'TT': 'TT', 'MSMA*': 'MSMA型', 'DiffPath': 'DiffPath', 'GEPC': 'GEPC', 'EigenScore': 'EigenScore', 'CFS': 'CFS', 'CFS-kNN': 'CFS-kNN'}
CITE = {'NLL': 'song2021score', 'IC-png': 'serra2020input', 'IC-j2k': 'serra2020input', 'TT': 'nalisnick2019typ', 'MSMA*': 'mahmood2021msma', 'DiffPath': 'heng2024diffpath',
        'GEPC': 'alexis2026gepc', 'EigenScore': 'shoushtari2026eigenscore', 'RotCls4': 'hendrycks2019ss', 'CFS': 'rouzoumka2026cfs', 'CFS-kNN': 'rouzoumka2026cfs'}
PASSES = {'NLL': '128+128', 'IC-png': '128+128', 'IC-j2k': '128+128', 'TT': '128+128', 'MSMA*': '20', 'DiffPath': '10', 'GEPC': '14', 'EigenScore': '180', 'RotCls4': '4', 'kNN (CT)': '1', 'CFS': '1', 'CFS-kNN': '1'}


def rd(v, n=1): return str(Decimal(repr(float(v))).quantize(Decimal(1).scaleb(-n), rounding=ROUND_HALF_UP))
def ex(sp, row, field, i=None):                              # the exact value where the two-decimal form ends in 5
    key = f'{sp}|{row}|{field}' + ('' if i is None else f'|{i}')
    if key in FIX: return FIX[key]
    o = S[sp]['ours'] if row == 'ours' else S[sp]['rows'][row]; v = o[field]; return v if i is None else v[i]
def f1(sp, row, field, i=None): return rd(ex(sp, row, field, i))
def five(sp, pair, i=0):
    key = f'{sp}|five|{pair}|{i}'; return FIX[key] if key in FIX else S[sp]['five'][pair][i]
def sg(v): s = rd(abs(v)); return '0.0' if s == '0.0' else ('$+$' if v > 0 else '$-$') + s
def wtl(sp, row): w = S[sp]['rows'][row]['wtl']; return f'{w[0]}/{w[1]}/{w[2]}'
W = {}                                                       # file name -> text
NUM = {}
def num(name, val): assert name.isalpha() and name not in NUM, name; NUM[name] = val

# ---------------------------------------------------------------- numbers of the text
for sp, q in (('test2', 'T'), ('testc', 'C')):
    num(f'Ours{q}', f1(sp, 'ours', 'mean')); num(f'OursLo{q}', rd(S['boot_abs'][sp]['E2'][0])); num(f'OursHi{q}', rd(S['boot_abs'][sp]['E2'][1]))
    num(f'Ud{q}', f1(sp, 'ours', 'vflip_mean')); num(f'Hf{q}', f1(sp, 'ours', 'hflip_mean')); num(f'Fpr{q}', f1(sp, 'ours', 'fpr_mean'))
    for nm, r in (('Prior', 'CFS-kNN'), ('Diag', 'CFS'), ('Knn', 'kNN (CT)'), ('Trained', 'A trained'), ('StopU', 'E1'), ('StopA', 'Ea'), ('OnlyJf', 'Jf'), ('OnlyJz', 'JzX'), ('Dens', 'Tf'), ('TfJz', 'Tf+JzX'), ('NoOri', 'Tf+TzX'), ('Four', 'four')):
        num(f'{nm}{q}', f1(sp, r, 'mean')); num(f'{nm}Ud{q}', f1(sp, r, 'vflip_mean')); num(f'{nm}Fpr{q}', f1(sp, r, 'fpr_mean')); num(f'{nm}Wtl{q}', wtl(sp, r))
        if S[sp]['rows'][r]['hflip_mean'] is not None: num(f'{nm}Hf{q}', f1(sp, r, 'hflip_mean'))
    for nm, pr in (('GapPrior', 'U|CFS-kNN'), ('GapDiag', 'U|CFS'), ('GapStopUPrior', 'E1|CFS-kNN'), ('GapTrained', 'U|A trained'), ('GapStopU', 'U|E1'), ('GapNoOri', 'U|Tf+TzX'), ('GapDens', 'U|K'), ('GapFour', 'U|four'), ('GapOnlyJf', 'U|Jf')):
        num(f'{nm}{q}', rd(abs(five(sp, pr, 0)))); num(f'{nm}Lo{q}', rd(abs(five(sp, pr, 1)) if nm == 'GapFour' else five(sp, pr, 1))); num(f'{nm}Hi{q}', rd(abs(five(sp, pr, 2)) if nm == 'GapFour' else five(sp, pr, 2)))
    for i, t in enumerate(TAGS):
        s = {'edmc10': 'Cten', 'l6svhn': 'Svhn', 'dpceleba': 'Celeb', 'in64': 'In', 'fmnist': 'Fm'}[t]
        num(f'Ours{s}{q}', f1(sp, 'ours', 'ids', i)); num(f'Trained{s}{q}', f1(sp, 'A trained', 'ids', i)); num(f'Prior{s}{q}', f1(sp, 'CFS-kNN', 'ids', i))
        num(f'Ud{s}{q}', f1(sp, 'ours', 'vflip', i)); num(f'UdTrained{s}{q}', f1(sp, 'A trained', 'vflip', i)); num(f'Hf{s}{q}', f1(sp, 'ours', 'hflip', i)); num(f'Fpr{s}{q}', f1(sp, 'ours', 'fpr', i))
num('RotT', f1('test2', 'RotCls4', 'mean')); num('RotUdT', f1('test2', 'RotCls4', 'vflip_mean')); num('EigenUdT', f1('test2', 'EigenScore', 'vflip_mean')); num('BasicT', f1('test2', 'kNN (CT)', 'mean'))
num('BestStatT', rd(max(ex('test2', r, 'mean') for r in PRIOR[:8]))); num('WtlPriorBest', wtl('test2', 'CFS-kNN'))
num('GapKnnT', rd(ex('test2', 'ours', 'mean') - ex('test2', 'kNN (CT)', 'mean')))                    # difference of the two means (no interval)
for i, s in enumerate(('Cten', 'Svhn', 'Celeb', 'In', 'Fm')): num(f'PriorBest{s}T', rd(max(ex('test2', r, 'ids', i) for r in PRIOR)))   # the best prior method per ID dataset (test)
# losses of Ours against a prior method, per pair (test)
LOSS = [(t, o, e['b'][1]) for t in TAGS for o, e in P['test2'][t].items() if o != '__hflip' and e['b'][2] == 'l']
num('NLossPairs', str(len(LOSS))); num('NLossSvhn', str(sum(1 for t, o, b in LOSS if t == 'l6svhn')))
# single pairs quoted in the text
def pv(sp, t, o, col=None): e = P[sp][t][o]; return rd(e['o'] if col is None else (e['b'][0] if col == 'b' else e['c'][col][0]))
num('PairCHundT', pv('test2', 'edmc10', 'cifar100_test')); num('PairCHundTrainedT', pv('test2', 'edmc10', 'cifar100_test', 2)); num('PairCHundPriorT', pv('test2', 'edmc10', 'cifar100_test', 'b'))
num('PairCCelT', pv('test2', 'edmc10', 'celeba_test')); num('PairCCelTrainedT', pv('test2', 'edmc10', 'celeba_test', 2))
num('PairSsbT', pv('test2', 'in64', 'in64_ssb_hard')); num('PairSsbTrainedT', pv('test2', 'in64', 'in64_ssb_hard', 2))
num('PairPlT', pv('test2', 'in64', 'in64_places365')); num('PairNincoT', pv('test2', 'in64', 'in64_ninco'))
num('PairInatT', pv('test2', 'in64', 'in64_inaturalist')); num('PairInatPriorT', pv('test2', 'in64', 'in64_inaturalist', 'b'))
num('PairSvhnNoiseT', pv('test2', 'l6svhn', 'synth_noise'))
# time (GPU, ms per image, batch 128)
B = S['bench']; sr = lambda t: B[t]['f_u'] + B[t]['f_o'] + B[t]['z_u'] + B[t]['z_o']; mean = lambda v: sum(v) / len(v)
# the forward pass up to the stop block / one full pass (blocks only: code/sx_bench2.py); everything (descriptors, searches) / one full pass
num('PassX', rd(mean([PASS[t]['X'] for t in TAGS]), 2)); num('PassA', rd(mean([PASS[t]['A'] for t in TAGS]), 2)); num('PassU', rd(mean([PASS[t]['U'] for t in TAGS]), 2))
num('PassXMin', rd(min(PASS[t]['X'] for t in TAGS), 2)); num('PassXMax', rd(max(PASS[t]['X'] for t in TAGS), 2))
num('AllX', rd(mean([B[t]['ours_X'] / B[t]['full'] for t in TAGS]), 2)); num('AllA', rd(mean([B[t]['ours_A'] / B[t]['full'] for t in TAGS]), 2)); num('AllU', rd(mean([B[t]['ours_U'] / B[t]['full'] for t in TAGS]), 2)); num('AllKnn', rd(mean([B[t]['knn_full'] / B[t]['full'] for t in TAGS]), 2))
num('AllXMin', rd(min(B[t]['ours_X'] / B[t]['full'] for t in TAGS), 2)); num('AllXMax', rd(max(B[t]['ours_X'] / B[t]['full'] for t in TAGS), 2))
CFSPASS = mean([PASS['cfs']['cifar10']['ratio_X'], PASS['cfs']['svhn']['ratio_X'], 1.0])   # CelebA: the stop block is the last block, counted as one full pass
num('CfsPass', rd(CFSPASS, 2)); num('CfsPassCten', rd(PASS['cfs']['cifar10']['ratio_X'], 2))
num('SearchShareMax', rd(100 * max(sr(t) / B[t]['ours_X'] for t in TAGS), 0)); num('SearchShareMin', rd(100 * min(sr(t) / B[t]['ours_X'] for t in TAGS), 0))
num('DimFMin', rd(min(S['dims'][t]['f'] for t in TAGS), 0)); num('DimFMax', rd(max(S['dims'][t]['f'] for t in TAGS), 0)); num('DimZMin', rd(min(S['dims'][t]['zX'] for t in TAGS), 0)); num('DimZMax', rd(max(S['dims'][t]['zX'] for t in TAGS), 0))
num('NzMin', str(min(S['blocks'][t][2] + 1 for t in TAGS))); num('NzMax', str(max(S['blocks'][t][2] + 1 for t in TAGS)))

# ---------------------------------------------------------------- Table 1: comparison on test
COLS = [('ids', 0), ('ids', 1), ('ids', 2), ('ids', 3), ('ids', 4), ('mean', None), ('vflip_mean', None), ('fpr_mean', None)]
def col(row, c): return float(ex('test2', row, c[0], c[1]))
best = {c: (min if c[0] == 'fpr_mean' else max)(float(rd(col(r, c))) for r in PRIOR) for c in COLS}
def cell(row, c, mark):
    s = rd(col(row, c)); v = float(s)
    if row == 'ours':
        lower = c[0] == 'fpr_mean'; better = sorted({float(rd(col(r, c))) for r in PRIOR}, reverse=not lower)
        if (v <= best[c]) if lower else (v >= best[c]): return f'\\cellcolor{{firstblue}}\\textbf{{{s}}}'
        if len(better) > 1 and ((v <= better[1]) if lower else (v >= better[1])): return f'\\underline{{{s}}}'
        return s
    return s + ('$^{\\dagger}$' if mark and v == best[c] else '')
L = ['% written by gen/mk.py: AUROC (%) on test; wins / ties / losses of Ours over the 51 pairs']
for r in PRIOR: L.append(f'{LABEL[r]}~\\cite{{{CITE[r]}}} & ' + ' & '.join(cell(r, c, True) for c in COLS) + f' & {wtl("test2", r)} & {PASSES[r]}\\\\')
L.append('\\midrule'); r = 'RotCls4'; L.append(f'{LABEL[r]}~\\cite{{{CITE[r]}}} & ' + ' & '.join(cell(r, c, False) for c in COLS) + f' & {wtl("test2", r)} & {PASSES[r]}\\\\')
L.append('\\midrule'); L.append('\\textbf{Ours：\\method} & ' + ' & '.join(cell('ours', c, False) for c in COLS) + f' & -- & {NUM["PassX"]}$^{{\\ddagger}}$\\\\\\bottomrule'); W['tab_main.tex'] = '\n'.join(L) + '\n'

# ---------------------------------------------------------------- Table 2: the parts and where the pass stops (test); the same rows on test' for the supplement
def prow(sp, label, r, time=None, last=''):
    g = lambda f: f1(sp, 'ours' if r is None else r, f)
    return f"{label} & {g('mean')} & {g('vflip_mean')} & {g('hflip_mean')} & {g('fpr_mean')} & {'--' if r is None else wtl(sp, r)}" + ('' if time is None else f' & {time}') + f"\\\\{last}"
PX, PA, PU = NUM['PassX'], NUM['PassA'], NUM['PassU']       # time of the pass up to the stop block / time of one full pass
ROWS2 = [('\\textbf{\\method}\\,($J_f+J_z$)', None, PX + '$^{\\ddagger}$', '\\midrule'), ('$J_f$ だけ', 'Jf', PU, ''), ('$J_z$ だけ', 'JzX', PX, ''), ('$T_f+J_z$', 'Tf+JzX', PX, ''),
         ('$T_f+T_z$\\,(比を使わない)', 'Tf+TzX', PX, ''), ('$T_f$\\,(U の密度だけ)', 'Tf', PU, ''), ('$T_f+Q_f+T_z+Q_z$', 'four', PX, '\\midrule'),
         ('U で止める', 'E1', PU, ''), ('2倍の解像度の最初で止める', 'Ea', PA, '\\midrule'), ('参考(学習あり)', 'A trained', '1', '')]
W['tab_parts.tex'] = '\n'.join(['% written by gen/mk.py: five-dataset means on test (AUROC %, FPR95 %); wins / ties / losses of Ours over the 51 pairs; time of the pass / one full pass']
                               + [prow('test2', lab, r, tm, last) for lab, r, tm, last in ROWS2]) + '\\bottomrule\n'
W['supp_parts_c.tex'] = '\n'.join(["% written by gen/mk.py: the rows of Table 2 on test'"] + [prow('testc', lab.replace('$^{\\ddagger}$', ''), r, None, last) for lab, r, tm, last in ROWS2]
                                  + ['\\midrule', prow('testc', 'CFS(kNN ヘッド)~\\cite{rouzoumka2026cfs}', 'CFS-kNN'), prow('testc', 'CFS(対角ガウス)~\\cite{rouzoumka2026cfs}', 'CFS', None, '\\bottomrule')]) + '\n'

# ---------------------------------------------------------------- Table 4: the evaluation procedure of CFS (published values + ours)
PUB = [('MSMA~\\cite{mahmood2021msma}', [66.3, 49.0, 60.6, 79.6, 93.7, 96.2, 93.6, 94.1, 64.0, 93.3, 73.8, 90.8], '79.6', '68.9', '10'),
       ('DiffPath~\\cite{heng2024diffpath}', [83.2, 31.6, 56.1, 78.0, 90.3, 96.0, 89.1, 92.4, 69.7, 97.0, 76.3, 91.1], '79.2', '63.5', '10'),
       ('DDPM-OOD~\\cite{graham2023ddpmood}', [12.4, 58.6, 56.8, 43.1, 90.7, 96.5, 92.6, 80.2, 44.1, 3.5, 51.6, 40.9], '55.9', '32.0', '364'),
       ('GEPC~\\cite{alexis2026gepc}', [91.7, 31.7, 50.6, 68.0, 96.3, 95.4, 85.5, 79.8, 79.9, 96.2, 68.6, 84.8], '77.4', '60.0', '8'),
       ('CFS$_{\\mathrm{dec}}$($1{\\times}1$)~\\cite{rouzoumka2026cfs}', [96.9, 45.1, 69.7, 96.8, 99.7, 99.3, 98.6, 99.1, 99.3, 99.9, 99.0, 99.9], '91.9', '80.9', '1'),
       ('CFS($1{\\times}2$)~\\cite{rouzoumka2026cfs}', [96.9, 46.8, 66.5, 95.9, 99.6, 99.0, 98.9, 99.1, 98.8, 99.9, 98.6, 99.8], '91.6', '81.4', '1'),
       ('SCOPED~\\cite{barkley2026scoped}', [81.4, 94.0, 47.7, None, 97.1, 99.6, 95.9, None, 92.5, 99.4, 96.2, None], '89.2$^{*}$', '--', '2+2')]
E1P = [('cifar10', o) for o in ('svhn', 'celeba', 'cifar100', 'dtd')] + [('svhn', o) for o in ('cifar10', 'celeba', 'cifar100', 'dtd')] + [('celeba', o) for o in ('cifar10', 'svhn', 'cifar100', 'dtd')]
if True:                                                     # the 12 pairs of the evaluation procedure of CFS (data/e1b_an.json)
    def e1row(r): rr = E1['rows'][r]; return [100 * rr['pairs'][f'{i}>{o}'] for i, o in E1P], 100 * rr['avg'], 100 * rr['avg_worst']
    bp = [max(v[k] for _, v, _, _, _ in PUB if v[k] is not None) for k in range(12)]; ba = max(float(a.replace('$^{*}$', '')) for _, _, a, _, _ in PUB); bw = max(float(w) for _, _, _, w, _ in PUB if w != '--')
    L = ['% written by gen/mk.py: the 12 pairs of the evaluation procedure of CFS (published values; ours: three noise seeds)']
    for lab, v, a, w, n in PUB:
        L.append(lab + ' & ' + ' & '.join('--' if x is None else (rd(x) + ('$^{\\dagger}$' if x == bp[k] else '')) for k, x in enumerate(v)) + f" & {a}{'$^{{\\dagger}}$' if a == rd(ba) else ''} & {w}{'$^{{\\dagger}}$' if w == rd(bw) else ''} & {n}\\\\")
    L.append('\\midrule')
    def orow(lab, r, n, bold):
        v, a, w = e1row(r); c = lambda x, b: (f'\\cellcolor{{firstblue}}\\textbf{{{rd(x)}}}' if bold and float(rd(x)) >= b else rd(x)); return lab + ' & ' + ' & '.join(c(x, bp[k]) for k, x in enumerate(v)) + f' & {c(a, ba)} & {c(w, bw)} & {n}\\\\'
    L.append(orow('参考(学習あり)', 'A trained (stored)', '1', False)); L.append(orow('\\textbf{Ours：\\method}', 'E2', NUM['CfsPass'] + '$^{\\ddagger}$', True) + '\\bottomrule'); W['tab_cfs.tex'] = '\n'.join(L) + '\n'
    v, a, w = e1row('E2'); num('CfsAvg', rd(a)); num('CfsWorst', rd(w)); num('CfsGap', rd(a - ba)); num('CfsGapWorst', rd(w - bw)); num('CfsBestAvg', rd(ba)); num('CfsBestWorst', rd(bw))
    num('CfsNWin', str(sum(1 for k, x in enumerate(v) if float(rd(x)) >= bp[k]))); va, aa, wa = e1row('A trained (stored)'); num('CfsTrainedAvg', rd(aa)); num('CfsTrainedWorst', rd(wa))
    num('CfsLo', rd(100 * E1['boot_avg_seed0'][0])); num('CfsHi', rd(100 * E1['boot_avg_seed0'][1]))
    for nm, (i, o) in (('CfsCCel', ('cifar10', 'celeba')), ('CfsCHund', ('cifar10', 'cifar100'))):
        k = E1P.index((i, o)); num(nm, rd(v[k])); num(nm + 'Trained', rd(va[k])); num(nm + 'Pub', rd(bp[k]))
    v1, a1, w1 = e1row('E1'); num('CfsStopU', rd(a1)); num('CfsDens', rd(e1row('Tf')[1]))

# ---------------------------------------------------------------- supplement: configuration per ID dataset
MODEL = {'edmc10': ('EDM(VP, DDPM++)~\\cite{karras2022edm}', 34, 'デコーダ $8{\\times}8$', '85.4'), 'l6svhn': ('VP-SDE~\\cite{kamkari2024geometric}', 13, 'デコーダ $8{\\times}8$', '82.7'),
         'dpceleba': ('iDDPM~\\cite{nichol2021improved}', 33, 'デコーダ $8{\\times}8$', '92.5'), 'in64': ('EDM2-XS(条件なし)~\\cite{karras2024edm2}', 37, 'デコーダ $8{\\times}8$', '72.2'),
         'fmnist': ('U-Net(EDMの前処理), 本研究で学習', 29, 'エンコーダ $8{\\times}8$', '93.8')}
NTR = {'edmc10': '5万', 'l6svhn': '5万', 'dpceleba': '5万', 'in64': '1万', 'fmnist': '5万'}; RES = {'edmc10': '$32{\\times}32$', 'l6svhn': '$32{\\times}32$', 'dpceleba': '$32{\\times}32$', 'in64': '$64{\\times}64$', 'fmnist': '$32{\\times}32$'}
L = ['% written by gen/mk.py']
for t in TAGS:
    m, nb, part, a_ = MODEL[t]; u, a, x = S['blocks'][t]; d = S['dims'][t]
    L.append(f"{NAME[t]} & {RES[t]} & {m} & {NTR[t]} & {nb} & {u + 1}({part}) & {a_} & {x + 1} & {rd(d['f'])} & {rd(d['zX'])}\\\\" + ('\\bottomrule' if t == TAGS[-1] else ''))
W['supp_models.tex'] = '\n'.join(L) + '\n'

# ---------------------------------------------------------------- supplement: every pair (test, test')
def pairs_table(sp, cols, with_best):
    L = ['% written by gen/mk.py']; pre = IDD
    for t in TAGS:
        order = NAT[t] + [k for k in P[sp][t] if 'synth' in k] + [IDD[t] + '__vflip', '__hflip']
        for j, o in enumerate(order):
            e = P[sp][t][o]; name = '上下反転' if o.endswith('__vflip') else ('左右反転$^{\\S}$' if o == '__hflip' else SETN[o])
            def cc(x, v): return ('\\cellcolor{firstblue}' if v == 'w' else ('\\cellcolor{lossred}' if v == 'l' else '')) + rd(x)
            cells = [rd(e['o'])] + [cc(*e['c'][k]) for k in cols]
            if with_best: cells.append((cc(e['b'][0], e['b'][2]) + f"\\,{{\\tiny {SHORT[e['b'][1]]}}}") if 'b' in e else '--')
            L.append((f'\\multirow{{{len(order)}}}{{*}}{{{NAME[t]}}}' if j == 0 else '') + f' & {name} & ' + ' & '.join(cells) + '\\\\')
        L.append('\\bottomrule' if t == TAGS[-1] else '\\midrule')
    return '\n'.join(L) + '\n'
W['supp_pairs_t.tex'] = pairs_table('test2', [3], True); W['supp_pairs_c.tex'] = pairs_table('testc', [1, 3], False)   # columns of the pair files: 1 CFS-kNN, 2 the trained reference, 3 stop at U

# ---------------------------------------------------------------- supplement: time per image
L = ['% written by gen/mk.py: GPU, batch 128, ms per image']
for t in TAGS:
    b = B[t]; L.append(f"{NAME[t]} & {rd(b['full'], 2)} & {rd(b['to_U'], 2)} & {rd(b['to_X'], 2)} & {rd(b['f_u'] + b['f_o'], 3)} & {rd(b['z_u'] + b['z_o'], 3)} & {rd(b['ours_X'], 2)} & {rd(PASS[t]['U'], 2)} & {rd(PASS[t]['X'], 2)} & {rd(b['ours_U'] / b['full'], 2)} & {rd(b['ours_X'] / b['full'], 2)}\\\\" + ('\\bottomrule' if t == TAGS[-1] else ''))
W['supp_time.tex'] = '\n'.join(L) + '\n'

# ---------------------------------------------------------------- Figure: amplitude profiles (pgfplots)
def coords(v): return ' '.join(f'({i + 1},{0 if x is None else x})' for i, x in enumerate(v))   # None: the three 1x1 blocks of the SVHN model, where the amplitude is the same number for every image (no deviation from the train mean: drawn as 0)
L = ['% written by gen/mk.py from data/fig_small.json (test images, noise seed 0)', '\\begin{tikzpicture}', '\\begin{groupplot}[group style={group size=5 by 1, horizontal sep=8mm}, width=41mm, height=32mm, unbounded coords=jump,',
     '  tick label style={font=\\figlab}, title style={font=\\scriptsize, yshift=-1.5mm, text height=1.6ex, text depth=0.3ex}, xlabel style={font=\\figlab}, ylabel style={font=\\figlab}, xlabel={ブロック(入力側 $\\rightarrow$ 出力側)}, every axis plot/.append style={line width=0.7pt},',
     '  legend style={font=\\figlab, draw=none, fill=none, legend columns=4, /tikz/every even column/.append style={column sep=3mm}}]']
for k, t in enumerate(TAGS):
    p = FG['profile'][t]; u, a, x = p['b']
    L.append(f"\\nextgroupplot[title={{{NAME[t]}}}" + (', ylabel={振幅(標準偏差を単位)}, legend to name=profleg' if k == 0 else '') + ']')
    L.append(f'\\draw[navyc!70, densely dotted, line width=0.6pt] (axis cs:{u + 1},\\pgfkeysvalueof{{/pgfplots/ymin}}) -- (axis cs:{u + 1},\\pgfkeysvalueof{{/pgfplots/ymax}}) node[pos=0.93, right, inner sep=1pt, font=\\figlab, text=navyc] {{U}};')
    L.append(f'\\draw[redc!80, densely dotted, line width=0.6pt] (axis cs:{x + 1},\\pgfkeysvalueof{{/pgfplots/ymin}}) -- (axis cs:{x + 1},\\pgfkeysvalueof{{/pgfplots/ymax}}) node[pos=0.78, left, inner sep=1pt, font=\\figlab, text=redc] {{X}};')
    L.append(f"\\addplot[navyc] coordinates {{{coords(p['id'])}}};"); L.append(f"\\addplot[redc] coordinates {{{coords(p['vf'])}}};")
    L.append(f"\\addplot[redc, densely dashed] coordinates {{{coords(p['oth'])}}};"); L.append(f"\\addplot[black] coordinates {{{coords(p['ood'])}}};")
    if k == 0: L.append('\\legend{分布内の画像, 上下反転した分布内の画像, 向きを変えた train 画像(7 通り), 自然画像の分布外集合(平均)}')
L += ['\\end{groupplot}', '\\end{tikzpicture}\\\\[-0.5mm]', '\\pgfplotslegendfromname{profleg}']; W['figs/fig_profile.tex'] = '\n'.join(L) + '\n'

# ---------------------------------------------------------------- Figure: the two readings of each descriptor (pgfplots)
def pts(v): return ' '.join(f'({x},{y})' for x, y in v)
L = ['% written by gen/mk.py from data/fig_small.json (CIFAR-10, test images, noise seed 0; 70 images per group)', '\\begin{tikzpicture}',
     '\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=9mm}, width=46mm, height=44mm, tick label style={font=\\figlab}, title style={font=\\scriptsize, yshift=-1.5mm, text height=1.6ex, text depth=0.3ex}, xlabel style={font=\\figlab}, ylabel style={font=\\figlab},',
     '  legend style={font=\\figlab, draw=none, fill=none, legend columns=3, /tikz/every even column/.append style={column sep=2mm}}]']
for k, (g, ttl) in enumerate((('f', '$f$：正立ブロックの平均と標準偏差'), ('z', '$z$：振幅の並び'))):
    s = FG['scatter'][g]; L.append(f"\\nextgroupplot[title={{{ttl}}}, xlabel={{密度の読み $T$}}" + (', ylabel={比の読み $Q$}, legend to name=scatleg' if k == 0 else '') + ']')
    L.append(f"\\addplot[only marks, mark=o, mark size=0.9pt, black!55, line width=0.3pt] coordinates {{{pts(s['ood'])}}};")
    L.append(f"\\addplot[only marks, mark=triangle*, mark size=1.1pt, redc] coordinates {{{pts(s['vf'])}}};")
    L.append(f"\\addplot[only marks, mark=*, mark size=0.9pt, navyc] coordinates {{{pts(s['id'])}}};")
    if k == 0: L.append('\\legend{自然画像の分布外集合, 上下反転した分布内の画像, 分布内の画像}')
L += ['\\end{groupplot}', '\\end{tikzpicture}\\\\[-0.5mm]', '\\pgfplotslegendfromname{scatleg}']; W['figs/fig_scatter.tex'] = '\n'.join(L) + '\n'

# ---------------------------------------------------------------- 256x256 models (supplement): data/hr256.json
HR = json.load(open(f'{D}/hr256.json')); sg = lambda v, n=1: rd(v, n)                       # signed values are set in math mode in the text
for t, q in (('lsb256', 'Lsb'), ('adm256c', 'Adm')):
    h = HR[t]; r = h['rows']; c = h['ci']
    num(f'Hr{q}', rd(r['E2'][0])); num(f'Hr{q}Lo', rd(c['E2'][0])); num(f'Hr{q}Hi', rd(c['E2'][1])); num(f'Hr{q}Ud', rd(r['E2'][1])); num(f'Hr{q}Hf', rd(r['E2'][2])); num(f'Hr{q}Fpr', rd(r['E2'][3]))
    num(f'Hr{q}Knn', rd(r['Tf'][0])); num(f'Hr{q}KnnUd', rd(r['Tf'][1])); num(f'Hr{q}Dens', rd(r['TfTz'][0])); num(f'Hr{q}DensUd', rd(r['TfTz'][1])); num(f'Hr{q}StopU', rd(r['E1'][0])); num(f'Hr{q}Jf', rd(r['Jf'][0]))
    num(f'Hr{q}GapKnn', sg(r['E2'][0] - r['Tf'][0])); num(f'Hr{q}GapKnnLo', sg(c['E2-Tf'][0])); num(f'Hr{q}GapKnnHi', sg(c['E2-Tf'][1]))
    num(f'Hr{q}GapDens', sg(r['E2'][0] - r['TfTz'][0], 2 if abs(r['E2'][0] - r['TfTz'][0]) < 0.5 else 1)); num(f'Hr{q}GapDensLo', sg(c['E2-TfTz'][0], 2 if abs(r['E2'][0] - r['TfTz'][0]) < 0.5 else 1)); num(f'Hr{q}GapDensHi', sg(c['E2-TfTz'][1], 2 if abs(r['E2'][0] - r['TfTz'][0]) < 0.5 else 1))
    num(f'Hr{q}GapStopU', sg(r['E2'][0] - r['E1'][0])); num(f'Hr{q}GapStopULo', sg(c['E2-E1'][0])); num(f'Hr{q}GapStopUHi', sg(c['E2-E1'][1]))
    pe = h['pairs']['E2']; num(f'Hr{q}PairMax', rd(max(pe.values()))); num(f'Hr{q}PairMin', rd(min(pe.values()))); num(f'Hr{q}A', rd(h['A'])); num(f'Hr{q}U', str(h['U'])); num(f'Hr{q}X', str(h['X'])); num(f'Hr{q}Blocks', str(h['nB']))
    num(f'Hr{q}Nid', str(h['n_id'])); num(f'Hr{q}DimF', rd(h['d']['f'])); num(f'Hr{q}DimZ', rd(h['d']['zX']))
num('HrLsbCfs', rd(HR['lsb256']['cfs']['diag'][0])); num('HrLsbCfsKnn', rd(HR['lsb256']['cfs']['knn'][0]))
A_ = HR['adm256c']; num('HrAdmCfs', rd(A_['cfs']['diag'][0])); num('HrAdmCfsKnn', rd(A_['cfs']['knn'][0])); num('HrAdmGapCfs', sg(A_['rows']['E2'][0] - A_['cfs']['diag'][0])); num('HrAdmGapCfsLo', sg(A_['ci']['E2-CFS_diag'][0])); num('HrAdmGapCfsHi', sg(A_['ci']['E2-CFS_diag'][1]))
num('HrAdmGapCfsKnn', sg(A_['rows']['E2'][0] - A_['cfs']['knn'][0])); num('HrLsbGapCfs', sg(HR['lsb256']['rows']['E2'][0] - HR['lsb256']['cfs']['diag'][0])); num('HrAdmCfsOpen', rd(A_['pairs']['CFS_diag']['openimage_o'])); num('HrAdmOpen', rd(A_['pairs']['E2']['openimage_o']))
num('HrSeeds', 'ノイズの種0' if HR['lsb256']['seeds'] == [0] else 'ノイズの種3つの平均'); assert HR['lsb256']['seeds'] == HR['adm256c']['seeds'] and HR['lsb256']['seeds'] in ([0], [0, 1, 2])
hrow = lambda lab, k: lab + ' & ' + ' & '.join(rd(v) for t in ('lsb256', 'adm256c') for v in HR[t]['rows'][k]) + '\\\\'
crow = lambda lab, k: lab + ' & ' + ' & '.join(rd(v) for t in ('lsb256', 'adm256c') for v in HR[t]['cfs'][k]) + '\\\\'
W['supp_hr.tex'] = '\n'.join(['% written by gen/mk.py: the 256x256 models (AUROC %, FPR95 %)', hrow('\\textbf{\\method}\\,($J_f+J_z$)', 'E2') + '\\midrule', hrow('$J_f$ だけ', 'Jf'), hrow('$J_z$ だけ', 'Jz'), hrow('$T_f+J_z$', 'TfJz'),
                              hrow('$T_f+T_z$\\,(比を使わない)', 'TfTz'), hrow('$T_f$\\,(U の密度だけ)', 'Tf'), hrow('$T_f+Q_f+T_z+Q_z$', 'four') + '\\midrule', hrow('U で止める', 'E1') + '\\midrule',
                              crow('CFS~\\cite{rouzoumka2026cfs}', 'diag'), crow('CFS(kNN ヘッド)~\\cite{rouzoumka2026cfs}', 'knn') + '\\bottomrule']) + '\n'

W['numbers.tex'] = '% written by gen/mk.py\n' + ''.join(f'\\newcommand{{\\n{k}}}{{{v}}}\n' for k, v in NUM.items())
for fn, txt in W.items(): open(os.path.join(OUT, fn), 'w', encoding='utf-8').write(txt)
print('written', len(W), 'files;', len(NUM), 'numbers; losses to a prior method (test):', LOSS)
for k in ('OursT', 'OursC', 'UdT', 'UdC', 'PriorT', 'GapPriorT', 'GapPriorLoT', 'GapPriorHiT', 'KnnT', 'GapKnnT', 'TrainedT', 'TrainedC', 'GapTrainedT', 'GapTrainedLoT', 'GapTrainedHiT', 'PassX', 'PassA', 'PassU', 'AllX', 'AllA', 'AllU', 'AllKnn', 'CfsPass', 'PassXMin', 'PassXMax', 'AllXMin', 'AllXMax', 'StopUT', 'StopUC', 'FourT', 'GapFourT', 'NLossPairs', 'NLossSvhn', 'SearchShareMin', 'SearchShareMax'): print(k, NUM[k], end=' | ')
print()
