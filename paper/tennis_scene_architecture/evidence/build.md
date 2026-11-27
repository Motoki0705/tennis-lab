# 初稿のビルド記録

2026-11-28に、正式レビュー第1回の修正を反映したLaTeX原稿と掲載図表で確認した。

| 項目 | 確認結果 |
| --- | --- |
| コマンド | `bash paper/tennis_scene_architecture/build.sh --check-tracked` |
| 終了値・経過 | 0、16.74秒 |
| PDF | A4、26ページ |
| 生成PDFのSHA-256 | `fbc053aec209354fb2d58ea1c4278f15dd1e5cecb74297c8fc75b20a4ca8aa4e` |
| 最終ログ | 未解決参照・引用0件、文字欠落0件、Overfull 0件、LaTeX・BibTeX警告0件 |
| Gitの入力照合 | 未追跡LaTeX入力0件 |
| 図表と参考文献 | 図14点、表17点、書誌18件 |
| 来歴 | 根拠49行、LaTeXの対応箇所527、106固定資料のSHA-256、コピー図版26ファイルを照合 |
| 再生成 | 保存データから生成した14表が一致。TikZの2図は再ビルド前後でバイト一致 |
| 付録 | マスク6種と理由配列3種の識別子をスキーマと照合。非ゼロ理由コードと実装名はroot 7種、関節6種、球7種で一致 |

LuaHBTeX 1.17.0（TeX Live 2023/Debian）、LuaTeX-ja 20231230.0、latexmk 4.83、BibTeX 0.99d、Poppler 24.02.0を使用した。本文の欧文・和文はDejaVuとNoto CJK JP、数式はLatin Modernを用いた。

参考文献、用語の参照先、図表番号の対応は組版時に照合した。初回組版では代表図表と付録を抜粋し、レビュー対応後は身体配置式、マスクと理由コード表、再生成手順のページを目視した。最終版の推敲と全ページの確認は [#2639](https://github.com/Motoki0705/tennis-lab/issues/2639) の工程に対応する。
