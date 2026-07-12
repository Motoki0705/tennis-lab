# paper/

英語論文（arXiv 投稿想定）のソース。`main.pdf` は `make` で生成する。

## ビルド

```bash
cd paper && make          # -> main.pdf
make clean                # 中間ファイル削除
make distclean            # main.pdf も削除
```

`latexmk` があれば自動でそちらを使い、`make watch` で保存時再ビルドできる。
無い場合は `pdflatex -> bibtex -> pdflatex x2` に自動フォールバックする。

必要なら: `sudo apt install latexmk`

## 構成

| パス | 役割 |
|---|---|
| `main.tex` | 章の `\input` と前付け・後付けのみ。本文は書かない |
| `preamble.tex` | パッケージ、記法マクロ、`\Todo`/`\Note` |
| `chapters/NN_*.tex` | 本文。`main.tex` の `\input` 順が章順 |
| `assets/figures/` | 図（`.pdf` 推奨、ラスタは `.png`） |
| `assets/tables/` | `\input` する表本体（`.tex`） |
| `refs.bib` | 参考文献。natbib + bibtex |

## 規約

- **記法はマクロ経由**。`preamble.tex` の `\ballThreeD` などを使い、生の記号を
  本文に直書きしない。付録の記法表と本文がずれるのを防ぐ。
- **図表は `assets/` に置き、キャプションで主張を述べる**。「何をしたか」ではなく
  「読者が何を見ているか」を書く。
- **未執筆箇所は `\Todo{...}` で明示する**。赤字で PDF に出る。camera-ready 時は
  `preamble.tex` の `\draftmodetrue` を `\draftmodefalse` にすれば全て消える
  （消えた場所が空白になるので、提出前に必ず全 `\Todo` を潰すこと）。
- **arXiv 互換**を保つ。`biblatex`/`biber` ではなく `natbib`+`bibtex`、
  `pdflatex` でビルドできる範囲のパッケージのみ使う。

## 環境メモ

この開発環境の TeX Live に入っていないもの: `latexmk`, `biber`, `biblatex`,
`IEEEtran`, `siunitx`, `algorithm2e`。
アルゴリズム擬似コードや SI 単位が必要になったら追加インストールが要る。
