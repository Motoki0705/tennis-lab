# コート検出・コートアライメント論文原稿

3DGS 再構成から作るテニスコート検出の学習データ生成と、
その前提になるメートル系コートアライメントをまとめた日本語原稿である。
LuaLaTeX と luatexja で組版し、中間物は `build/` に隔離する。

## 構成

| パス | 内容 |
|---|---|
| `main.tex` | 原稿の入口。本文・付録の `\input` と参考文献の指定 |
| `preamble.tex` | プリアンブル。配色、TikZ スタイル、表記マクロ |
| `sections/` | 本文 00--11 |
| `appendices/` | 付録 A--C（座標契約、閾値、再現性） |
| `tables/` | 表 |
| `figures/*.tex` | TikZ で描く図 |
| `figures/generated/` | 実測から生成した結果図（PNG） |
| `results/generated/benchmark_metrics.tex` | 数値の唯一の入力点 |
| `refs.bib` | 参考文献 |
| `Makefile`, `latexmkrc` | ビルド設定 |

## 依存

- `latexmk`、`lualatex`、`bibtex` を含む TeX Live
- LuaLaTeX 用の日本語組版（luatexja）と、
  geometry、amsmath、amssymb、bm、graphicx、booktabs、multirow、array、
  tabularx、enumitem、placeins、caption、subcaption、xcolor、xspace、
  tikz、natbib、hyperref、cleveref
- `latexmkrc` が `$pdf_mode = 4`（LuaLaTeX）、`$out_dir = build`、
  成功時の `build/main.pdf` → `main.pdf` コピーを設定する

## 使い方

| コマンド | 動作 |
|---|---|
| `make` | `main.pdf` を生成する |
| `make watch` | `latexmk -pvc` で監視し、変更のたびに再構築する |
| `make check` | ビルド後、未定義参照・引用、TeX エラー、欠落ファイル、40pt 超の Overfull を検査する |
| `make clean` | 中間物のみ削除する |
| `make distclean` | 中間物と `main.pdf` を削除する |

`make watch` は既定でビューアを開かない。
`make watch VIEW=pdf` のように `VIEW` を渡すと `latexmk -view` へ渡す。
`SOURCES` は `figures/generated/` の PNG も wildcard で含むため、
TeX を触らず図だけを差し替えた場合も再ビルドが走る。
監視中もビルドに成功した時点で `build/main.pdf` がトップレベルの
`main.pdf` へコピーされる。失敗時はコピーしないため、
古い PDF を新しい結果と取り違えない。

`main.pdf` はビルド生成物であり、原稿のソースではない。

## 数値と結果図の差し込み

本文・表・図は数値を直接書かない。
数値は `results/generated/benchmark_metrics.tex` で定義したマクロだけを参照する。

現在の版は測定済みである。
`results/generated/benchmark_metrics.tex` の
`\resultsmeasuredtrue` と `\resultsfigurestrue` は測定状態の記録であり、
本文・表・図は実ファイルと実測マクロを直接参照する。
未測定の項目だけが `\resultsUnmeasured` を返す。

結果図は次の 4 枚（および古典 baseline 補助 probe の 4 枚、うち本文使用 3 枚）で、
`knowledge/runs/run-court-alignment-crossmodel-n120-s42/figures/` から
複製したものである。

| ファイル | 内容 |
|---|---|
| `summary_bars.png` | 主要指標の棒グラフ（欠落込みの PCK / H success を含む） |
| `error_cdf.png` | 有効対応のみのキーポイント誤差 CDF（n はモデル間で異なる） |
| `montage_real_validation.png` | 実写 validation の代表 5 例（GT 緑 / ours 橙 / TCD 青 / 重畳） |
| `montage_synthetic_test.png` | 合成 test の代表 5 例（同上） |
| `classical/*.png` | 古典 baseline 補助 probe の代表例（図が使うのは 3 枚、残りは来歴記録） |

各 PNG の SHA-256 と生成元 run は
付録 C（`appendices/C_reproducibility.tex`）に記録する。

### 数値を更新する手順

1. 新しい run の `metrics.json` と突合し、
   `results/generated/benchmark_metrics.tex` のマクロを差し替える。
2. 図を `figures/generated/` へ置き換える。
3. `make distclean && make check` で参照・Overfull・欠落ファイルを確認する。
4. `pdftotext main.pdf -` で主要数値が本文に出ていることを確認する。

本文・表・図の側を編集して数値を直接書き込まない。

### 未実施として残すもの

次の項目は本稿でも未実施であり、「未測定」と明示する。
測定済みに見える表や図を作らない。

- `tables/ablation_plan.tex` の視点分布 ablation（A--E）
- 学習へ渡った視点分布（方位・高さ・距離）の集計図
- 実写のみで学習した同一アーキテクチャ比較、未知会場の分離、複数 seed
