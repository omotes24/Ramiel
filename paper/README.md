# paper/ — 論文(MIRU 2025 のクラス、pLaTeX)

## ビルド

    python3 gen/mk.py && python3 gen/mk_schem.py && latexmk -r latexmkrc main.tex

- 数値(`numbers.tex`)、表の本体(`tab_*.tex`, `supp_*.tex`)、データの図(`figs/fig_profile.tex`, `figs/fig_scatter.tex`)は、`gen/mk.py` が `data/` から作ります。手で書いた数値はありません。
- 模式図は `figs/fig_method.tex`, `figs/fig_idea.tex`(TikZ。点の配置は `gen/mk_schem.py`)。
- 手法の名前は `main.tex` の `\method` の 1 か所で変えられます。
- `main.pdf` は、このソースから組んだものです(フォントは原ノ味を埋め込み)。

## データ(`data/`、GPU サーバからの写し)

| ファイル | 中身 |
|---|---|
| `sum_main.json`, `sum_fix.json` | test と test′ の要約(手法ごと・分布内ごとの AUROC と FPR95、信頼区間、勝ち負け) |
| `sum_pairs2.json`, `sum_pairsc.json` | 組ごとの AUROC(test、test′) |
| `sum_pass.json` | 計算時間 |
| `e1b_an.json` | CFS の論文の評価手順(12 組) |
| `fig_small.json` | 図のためのデータ(振幅の並び、散布図) |
| `hr256.json` | 256×256 のモデルでの結果 |

サーバでの分割名は、test2 が本文の test、testc が test′ です。
