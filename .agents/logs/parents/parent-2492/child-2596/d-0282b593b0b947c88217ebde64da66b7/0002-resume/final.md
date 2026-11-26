人物モデル比較の5ページを GitHub Wiki に公開しました。`master` のコミットは `a13f1662d77c7bb7720aec562173f2fbe5ad7fbd` です。

1. Top-down pose の主表に PMPose-H の2行を戻し、補助表を ViTPose-H flipなしの2行に修正しました。主比較30条件と補助6条件の計時・VRAM、各行の OKS と PT32 差を出典に再照合しました。概要の比較表と候補表も確認しました。
2. 検出・姿勢・人体メッシュの3ページで、可視化と動画の保管に関する文をです・ます調にそろえました。
3. push 直前は #2593 が未マージ、#2491 がオープンだったため、概要には記録 PR で反映予定と記載しています。 [PR #2593](https://github.com/Motoki0705/tennis-lab/pull/2593)・[Issue #2491](https://github.com/Motoki0705/tennis-lab/issues/2491)

新規5ページはすべて HTTP 200 で本文を取得でき、Home・サイドバー・標準パイプライン入口からの導線も確認しました。禁止パターンの検出は0件です。 [概要](https://github.com/Motoki0705/tennis-lab/wiki/Person-Model-Comparison-Overview)・[比較方法](https://github.com/Motoki0705/tennis-lab/wiki/Person-Model-Comparison-Methods)・[人物検出と ECPose](https://github.com/Motoki0705/tennis-lab/wiki/Person-Detector-and-ECPose-Comparison)・[Top-down pose](https://github.com/Motoki0705/tennis-lab/wiki/Top-Down-Pose-Model-Comparison)・[FastHMR と GVHMR](https://github.com/Motoki0705/tennis-lab/wiki/Body-Recovery-FastHMR-vs-GVHMR)

再照合と公開結果は [照合記録](https://github.com/Motoki0705/tennis-lab/blob/052f43b1d83b2cc16bccc647c8fce2f60de4a45d/tmp/wiki-evidence.md) に追記しました。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}