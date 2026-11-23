#!/usr/bin/env python3
# gen/mk_schem.py -- the point sets and the circles of the two schematic figures: Figure 1 (c) and Figure 2.
#   Both are drawn for R = 1 and to the letter: the upright bank u has 6 entries (navy dots), the re-oriented bank o has 7 x 6 = 42
#   (grey squares); the solid circle passes through the nearest entry of u, the dashed circle through the 7th nearest entry of o.
#   The radii are computed here from the points, so the circles pass exactly through the entries that define them.
#   Writes figs/fig_method_c.tex (input by figs/fig_method.tex) and figs/fig_idea.tex.  No data files are read.
import math, os, random
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs')
def dist(a, b): return math.hypot(a[0] - b[0], a[1] - b[1])
def order(q, pts): return sorted(pts, key=lambda p: dist(q, p))
def P(p): return f'({p[0]:.3f},{p[1]:.3f})'
def pol(q, r, deg): return (q[0] + r * math.cos(math.radians(deg)), q[1] + r * math.sin(math.radians(deg)))
def check(q, navy, grey, what):
    n = order(q, navy); g = order(q, grey); ru = dist(q, n[0]); ro = dist(q, g[6])
    assert dist(q, n[1]) - ru > 0.05, (what, 'the nearest dot is not clearly the nearest')
    assert ro - dist(q, g[5]) > 0.03 and dist(q, g[7]) - ro > 0.03, (what, 'the 7th square is not clearly the 7th', dist(q, g[5]), ro, dist(q, g[7]))
    return n[0], g[6], ru, ro

# ---------------------------------------------------------------- Figure 1 (c): local coordinates, origin = top left corner of the panel
q = (2.15, -1.30)
navy = [pol(q, 0.58, 158), (1.38, -1.98), (1.12, -0.48), (0.78, -1.22), (0.38, -0.62), (0.36, -1.82)]
near = [pol(q, r, a) for r, a in ((0.88, 5), (0.97, -14), (1.00, 22), (1.08, -27), (1.06, 37), (1.10, 9))]   # the six squares inside the dashed circle
seventh = pol(q, 1.15, -38)                                                                                    # the square on the dashed circle
rng = random.Random(3)
cand = [(3.20 + 0.32 * i + rng.uniform(-0.06, 0.06), -0.30 - 0.352 * j + rng.uniform(-0.06, 0.06)) for i in range(6) for j in range(7)]
cand = [p for p in cand if dist(q, p) >= 1.27 and min(dist(p, s) for s in near + [seventh]) >= 0.17]
rng.shuffle(cand); far = sorted(cand[:35]); assert len(far) == 35, len(cand)
grey = near + [seventh] + far; assert len(grey) == 7 * len(navy)
n1, g7, ru, ro = check(q, navy, grey, 'figure 1c'); assert dist(g7, seventh) < 1e-9
L = ['% written by gen/mk_schem.py: the two banks of Figure 1 (c), drawn for R = 1 (6 navy dots = u, 42 grey squares = o)',
     f'\\foreach \\p in {{{",".join(P(p) for p in navy)}}} \\fill[navyc] \\p circle (0.055);',
     f'\\foreach \\p in {{{",".join(P(p) for p in grey)}}} \\fill[black!45] \\p ++(-0.04,-0.04) rectangle ++(0.08,0.08);',
     f'\\draw[navyc, line width=0.6pt] {P(q)} circle ({ru:.3f});',
     f'\\draw[black!60, densely dashed, line width=0.6pt] {P(q)} circle ({ro:.3f});',
     f'\\draw[navyc, line width=0.6pt] {P(q)} -- node[pos=0.62, above right, inner sep=1pt, text=navyc, font=\\figlab] {{$r^{{u}}_{{R}}$}} {P(n1)};',
     f'\\draw[black!65, line width=0.6pt] {P(q)} -- node[pos=0.80, below left, inner sep=0.5pt, text=black!75, font=\\figlab] {{$r^{{o}}_{{R}}$}} {P(g7)};',
     f'\\node[star, star points=5, star point ratio=2.2, fill=redc, inner sep=1.2pt] at {P(q)} {{}};']
open(os.path.join(OUT, 'fig_method_c.tex'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print(f'figure 1c: r_u {ru:.3f}, r_o {ro:.3f}; {len(navy)} dots, {len(grey)} squares')

# ---------------------------------------------------------------- Figure 2: three panels with the same two banks
W, H = 2.62, 2.30
navy = [(0.42, -0.50), (0.80, -0.58), (0.28, -0.98), (0.66, -1.12), (0.98, -0.95), (0.55, -1.52)]
stars = [(0.62, -0.86), (1.98, -0.95), (1.22, -2.03)]
for seed in range(1000):                                    # the first jitter of the grid for which the 7th square is clearly the 7th in all panels
    rng = random.Random(seed)
    grey = [(1.50 + 0.19 * i + rng.uniform(-0.045, 0.045), -0.32 - 0.21 * j + rng.uniform(-0.045, 0.045)) for i in range(6) for j in range(7)]
    try:
        for k, s in enumerate(stars): check(s, navy, grey, k)
        break
    except AssertionError: continue
else: raise SystemExit('no jitter found')
assert len(grey) == 7 * len(navy); print('figure 2: jitter seed', seed)
L = ['% written by gen/mk_schem.py -- Figure 2: what the two readings see (schematic, drawn for R = 1; natural width about 8.3 cm).',
     '%   The same two banks in three panels: 6 navy dots = the upright train images (u), 42 grey squares = the train images in the seven',
     '%   other orientations (o).  The input (red star): (i) an ID image, (ii) an image whose orientation alone differs, (iii) an image whose',
     '%   content differs.  The solid circle passes through the nearest entry of u, the dashed circle through the 7th nearest entry of o.',
     '\\begin{tikzpicture}[font=\\figlab, line join=round, line cap=round, lab/.style={font=\\figlab, inner sep=1pt, align=center}]']
for k, s in enumerate(stars):
    n1, g7, ru, ro = check(s, navy, grey, f'figure 2 panel {k}'); ox = 2.82 * k
    print(f'figure 2 ({"i" * (k + 1)}): r_u {ru:.3f}, r_o {ro:.3f}, r_o / r_u {ro / ru:.2f}')
    L += [f'\\begin{{scope}}[shift={{({ox:.2f},0)}}]', f'  \\draw[black!40, rounded corners=2pt] (0,0) rectangle ({W},-{H});',
          f'  \\foreach \\p in {{{",".join(P(p) for p in navy)}}} \\fill[navyc] \\p circle (0.045);',
          f'  \\foreach \\p in {{{",".join(P(p) for p in grey)}}} \\fill[black!45] \\p ++(-0.03,-0.03) rectangle ++(0.06,0.06);',
          f'  \\begin{{scope}} \\clip (0.01,-0.01) rectangle ({W - 0.01},-{H - 0.01});',
          f'    \\draw[navyc, line width=0.6pt] {P(s)} circle ({ru:.3f});', f'    \\draw[black!60, densely dashed, line width=0.6pt] {P(s)} circle ({ro:.3f});', '  \\end{scope}',
          f'  \\node[star, star points=5, star point ratio=2.2, fill=redc, inner sep=1.0pt] at {P(s)} {{}};', '\\end{scope}']
L += ['\\node[lab, anchor=south] at (1.31,0.05) {\\textbf{(i)}\\ 分布内の画像};', '\\node[lab, anchor=south] at (4.13,0.05) {\\textbf{(ii)}\\ 向きだけが異なる画像};',
      '\\node[lab, anchor=south] at (6.95,0.05) {\\textbf{(iii)}\\ 内容が異なる画像};',
      '\\node[lab, anchor=north, align=center] at (1.31,-2.36) {$u$ に近く, $o$ から遠い\\\\密度：高い,\\ 比：高い};',
      '\\node[lab, anchor=north, align=center] at (4.13,-2.36) {$u$ より $o$ に近い\\\\\\textcolor{redc}{比：低い}};',
      '\\node[lab, anchor=north, align=center] at (6.95,-2.36) {$u$ からも $o$ からも遠い\\\\\\textcolor{redc}{密度：低い}};',
      '\\fill[navyc] (0.12,-3.28) circle (0.045); \\node[lab, anchor=west, text=navyc] at (0.2,-3.28) {正立の train 画像($u$)};',
      '\\fill[black!45] (2.72,-3.31) rectangle ++(0.06,0.06); \\node[lab, anchor=west, text=black!75] at (2.84,-3.28) {向きを変えた train 画像($o$)};',
      '\\node[star, star points=5, star point ratio=2.2, fill=redc, inner sep=0.9pt] at (6.2,-3.28) {}; \\node[lab, anchor=west] at (6.3,-3.28) {入力};',
      '\\end{tikzpicture}']
open(os.path.join(OUT, 'fig_idea.tex'), 'w', encoding='utf-8').write('\n'.join(L) + '\n')
