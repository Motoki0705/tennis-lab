# 初稿のビルド記録

2026-11-28に、初稿のLaTeX文書と掲載図表をステージした状態で確認した。

| 項目 | 確認結果 |
| --- | --- |
| コマンド | `bash paper/tennis_scene_architecture/build.sh --check-tracked` |
| 終了値・経過 | 0、16.28秒 |
| PDF | A4、25ページ |
| 生成PDFのSHA-256 | `c1f691166d13b3f99f7496a6c15a02d9c251dd73334b1d83411b1c522c010817` |
| 最終ログ | 未解決参照・引用0件、文字欠落0件、Overfull 0件、LaTeX・BibTeX警告0件 |
| Gitの入力照合 | 未追跡LaTeX入力0件 |
| 図表と参考文献 | 図14点、表15点、書誌18件 |
| 来歴 | 根拠46行、LaTeXの対応箇所507、97固定資料のSHA-256、コピー図版26ファイルを照合 |
| 再生成 | 保存データから生成した14表が一致。TikZの2図は再ビルド前後でバイト一致 |

LuaHBTeX 1.17.0（TeX Live 2023/Debian）、LuaTeX-ja 20231230.0、latexmk 4.83、BibTeX 0.99d、Poppler 24.02.0を使用した。本文の欧文・和文はDejaVuとNoto CJK JP、数式はLatin Modernを用いた。

参考文献、用語の参照先、図表番号の対応は組版時に照合した。PDFは表題と要旨、パイプライン図、コート画像、人物の診断例、球像分布と時系列比較、実行システム図、性能比較、付録のページを抜粋して目視した。最終版の推敲と全ページの確認は [#2639](https://github.com/Motoki0705/tennis-lab/issues/2639) の工程に対応する。
