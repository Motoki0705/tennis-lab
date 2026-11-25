`codex/parent-2492-issue-2490-primary` の `1dda95b51398b15e9c69d0b4bf2a92ebe7472eeb` をpushし、実行証拠を更新しました。 [ドラフトPR #2500](https://github.com/Motoki0705/tennis-lab/pull/2500)

- 18条件で7群×17関節の誤差・PCK・欠測・jitterを集計し、全体値を共通採点器と照合しました。遠距離の代理にBBox高さ80 px未満、高速動作の代理に手首速度の上位10%を使いました。高速群の左手首jitter P95はViTPose-H PT32で33.75 px、PMPose-H PT32で29.04 pxでした。
- 6クリップ×4カメラの比較画像24枚を元フレームと照合しました。評価間隔が不規則な `clip_004` の動画と、手首誤差が最大だった例の拡大図も追加しました。 [集計と可視化の記録](https://github.com/Motoki0705/tennis-lab/blob/1dda95b51398b15e9c69d0b4bf2a92ebe7472eeb/knowledge/nodes/tennis_scene/000086-run-meiji-topdown-pose-comparison.md)
- Sapiens2のadapterと両サイズのONNX検証結果を統合しました。初回TensorRT buildは両サイズとも `ReduceMean` のaxes要件で停止しました。1B PT32は実行中、両サイズのPT16/TRT16精度ジョブは取消済みです。5B PT32はCUDAロードまで確認し、forwardと常駐VRAMは未確認です。

対象pytest 69件、forward境界テスト1件、pre-commit、knowledge検査が通過しました。新HEADのCI結果を待ちます。速度再計測は #2497 の修正版と親の再投入指示が揃ってから進めます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2500,"head":"1dda95b51398b15e9c69d0b4bf2a92ebe7472eeb","resume_on_success":true,"next_step":"Sapiens2担当の精度・変換の続報を照合して統合する。#2497の修正版と親の再投入指示が揃ったら、共通速度計測とPareto比較を進める。"}}