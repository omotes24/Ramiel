# code/ — BURST のコード(GPU サーバで実行したもの)

サーバ(4× RTX 2080 Ti)で実行したファイルの写しです。変えたのは、プロジェクトのルートと Python インタプリタのパスだけです(`../tools/subs.py`)。置き換える前の、実行した版の md5 は `md5_server.txt` にあります。

リポジトリを `~/ramiel` に置き、そこから実行します。`data/npy/<集合>.npy`(画像の配列)、`ckpt/`(モデルの重み)、`ext/`(外部のモデルのコード)を読み、`runs/` に書きます。タグは、分布内データセットとその拡散モデルの組です(`edmc10` CIFAR-10 / EDM、`l6svhn` SVHN / VP-SDE、`dpceleba` CelebA / iDDPM、`in64` ImageNet-64 / EDM2-XS、`fmnist` FashionMNIST / 本研究の U-Net、`lsb256` LSUN の寝室 / ADM、`adm256c` ImageNet 256 / ADM)。論文の test は集合名の `test2-`、test′ は `testc-` です。

## 共通

| ファイル | 内容 |
|---|---|
| `dgcore.py` | モデルの読み込み、画像の 8 通りの変換(`TF`)、共有のノイズ画像(`shared_noise`) |
| `ct_extract_defs.py` | タグごとのモデルとバンク(`SPEC`)、読むブロックの規則(`PATS`)、バンクの画像の選び方(`bank_indices`) |
| `admwrap.py`, `dgnet.py` | ADM 256×256 のラッパーと、FashionMNIST 用に学習した U-Net の定義 |

## ブロック U の選択(train 画像だけ)

| ファイル | 内容 |
|---|---|
| `up_extract.py` | すべてのブロックの統計量を、バンクの 1 万枚と、取り分けた train 画像 5,000 枚の 8 通りの向きについて取り出す |
| `knorm_depth.py`, `knorm_depth_rules.py` | ブロックごとの最近傍の距離と、そのままの画像と向きを変えた写しを見分ける AUROC。最大のブロックが U(規則 `max orientation`) |
| `knorm_up_extract.py` | U の平均と標準偏差を、バンクと各集合について取り出す(`--block` に U を渡す) |

## BURST(順伝播 1 回、学習なし)

| ファイル | 内容 |
|---|---|
| `sx_un_extract.py` | バンクの 8 通りの向きの特徴 $f$(U で止める順伝播) |
| `sx_ap_extract.py` | バンクの 8 通りの向きと各集合の振幅の並び(X で止める順伝播) |
| `sx_tv_extract.py` | test / test′ の各集合の特徴 $f$(列 0 を使う) |
| `sx_e_terms.py` | 画像ごとの項 $J_g$, $T_g$, $Q_g$(`ldist` が最近傍の距離、`terms` が式の本体) |
| `sx_e_eval.py` | 1 回だけの評価(test と test′、ノイズの種 0〜2)。スコアは `E2` の行 |
| `sx_e_bench.py` | 1 枚あたりの時間(順伝播を止める位置ごと、探索) |
| `sx_e1b.py`, `sx_e1b_an.py` | CFS の論文の評価手順(CIFAR-10 のモデル 1 つ、12 組) |

## 256×256 のモデルでの確認

| ファイル | 内容 |
|---|---|
| `tc3_feat.py`, `tc3_depth.py` | すべてのブロックの特徴と、ブロック U の選択 |
| `sx_g_extract.py`, `sx_g_run.sh` | バンクの 8 通りの向きと test の各集合の $f$ と振幅の並び(X で止める順伝播) |
| `sx_g_eval.py` | 評価(項の計算は `sx_e_terms.py` と同じ) |

## 入っていないもの

- データの準備(`data/npy/*.npy`)、集合の定義(`runs/*/sets_<タグ>.json`)、モデルの重み(`ckpt/`)、外部のコード(`ext/`)。
- 比較手法の実行と、保存したスコア。`sx_e_eval.py`, `sx_e1b.py` は、比較の行のためにそれらを読みます(`runs/ax/`, `runs/e11/`, `runs/orbc/`, `runs/cfs_l7/` など)。このため、評価のスクリプトはこのリポジトリだけでは最後まで動きません。集計した結果は `../paper/data/` にあります。
- 論文の表のための集計スクリプト。
