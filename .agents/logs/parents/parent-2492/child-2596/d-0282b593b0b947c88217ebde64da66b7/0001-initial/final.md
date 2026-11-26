#2596 の Wiki 原稿5ページと既存7ページの導線を `tmp/wiki/` に用意しました。親レビューと公開承認待ちです。

| 新規ページ | 行数 | 主な節と出典 |
|---|---:|---|
| [概要・結論](https://github.com/Motoki0705/tennis-lab/blob/052f43b1d83b2cc16bccc647c8fce2f60de4a45d/tmp/wiki/Person-Model-Comparison-Overview.md) | 121 | 段別の結論、ユーザー判断、親の推奨、下流確認。#2492・#2494 と各比較表 |
| [比較方法](https://github.com/Motoki0705/tennis-lab/blob/052f43b1d83b2cc16bccc647c8fce2f60de4a45d/tmp/wiki/Person-Model-Comparison-Methods.md) | 183 | 代表6クリップ、疑似GT、品質マスク、採点・計時・出力差。`publication.json`、`quality_counts.csv`、採点器・ハーネス |
| [人物検出・ECPose](https://github.com/Motoki0705/tennis-lab/blob/052f43b1d83b2cc16bccc647c8fce2f60de4a45d/tmp/wiki/Person-Detector-and-ECPose-Comparison.md) | 231 | 全44条件、変換不具合、別 recipe とキャッシュ。`conditions.csv` と各モデルの精度・計時 JSON |
| [Top-down 2D pose](https://github.com/Motoki0705/tennis-lab/blob/052f43b1d83b2cc16bccc647c8fce2f60de4a45d/tmp/wiki/Top-Down-Pose-Model-Comparison.md) | 169 | 主比較30条件、難条件、Sapiens2 の除外、採用前の確認。`comparison.csv`、`accuracy.csv`、`difficulty_pt32.csv` |
| [FastHMR と GVHMR](https://github.com/Motoki0705/tennis-lab/blob/052f43b1d83b2cc16bccc647c8fce2f60de4a45d/tmp/wiki/Body-Recovery-FastHMR-vs-GVHMR.md) | 127 | 構成・ライセンス、速度・VRAM、8フレームの出力差、見送り判断。`speed-summary.json`、現行 `cases.json`、`observations.json` |

既存の変更は `_Sidebar.md`、`Home.md`、`Tennis-Scene-Pipeline-Overview.md`、`Component-Person-Detection-and-Tracking.md`、`Component-3D-Reconstruction-and-Tooling.md`、`Tennis-Scene-History-and-Optimizations.md`、`Meiji-4cam-Benchmarks.md` の導線です。

表19件の数値、内部リンク47件、main の参照先80ファイルを照合しました。Pareto 図3点は公開 URL が HTTP 200 で、main の画像とバイト一致しました。追加内容の公開禁止パターンは0件です。各表の出典、検証コマンド、レビュー対象12ファイルの SHA-256 は [照合記録](https://github.com/Motoki0705/tennis-lab/blob/052f43b1d83b2cc16bccc647c8fce2f60de4a45d/tmp/wiki-evidence.md) にあります。

#2593 は確認時点で未マージのため、FastHMR の見送りは親の判断記録を出典にしました。Sapiens2-5B は保存記録どおり CUDA ロード完了とし、forward と常駐 VRAM を未計測と記載しました。#2487 の追跡・3Dへの影響、組み合わせた経路全体の速度、一部の重み・engine の利用条件は未確認です。 [FastHMR の判断記録](https://github.com/Motoki0705/tennis-lab/issues/2494#issuecomment-7178978237)

作業ブランチは `codex/parent-2492-issue-2596-primary`、Wiki の基点は `master` の `5a73e6c`、照合した main は `052f43b` です。コミット、push、新規5ページの公開 URL 確認は承認後に行います。

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}