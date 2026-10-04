# プレイ区間・32frame学習窓のレビュー

`play_intervals.py`は注釈からプレイ区間を提案する純粋関数。
推論時のプレイ検出器ではなく、固定済み学習データの選択にだけ使う。
設定値の正本は`PlayIntervalConfig`、実行値は生成manifestへ保存する。

1. レビュー済み・target frame・単一球のobserved/interpolated/occlusion_estimated/unresolvedを存在の証拠にする。
2. 前後に証拠がある短い欠損だけを実PTS秒で連結する。区間端を延長せず、reference-only frameと時刻の大きな飛びを跨がない。
3. 実frame数と存在証拠率を満たす連続区間をプレイ候補とし、残りを非プレイ候補とする。
4. 各候補内で固定長の窓を切り出す。位置教師が十分ある窓だけを学習に採用する。

`segment_break`はhit/bounce/cut/play boundaryを混在して表すため、一律に切断しない。
複数球・未レビュー・out_of_frameは存在の正証拠にしないが、短い内部欠損として
連結され得る。連結は区間選択だけを変え、座標教師や存在ラベルを書き換えない。

**プレイ候補と学習窓の被覆は別**。unresolved主体のプレイ区間は位置教師不足で
学習対象にならない場合がある。非プレイ候補は確定GTではなく、短いラリー・長い遮蔽・
画面外への移動を誤って除外し得る。逆に静止球や打球準備を含む可能性もある。

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.review_play_intervals \
  --poses <absolute-root>/ball-mix-v2-player-pose-v1 \
  --output <absolute-new-report-directory>
```

CPUのみ。学習・モデル推論はしない。`manifest.json`にapproved subset、ball snapshot、
pose artifact、学習窓と全除外範囲を固定し、`summary.json`・`timelines.png`・
`example-*.jpg`に集計と実画像を出す。サンプル選択はtrain clipだけで、長い非プレイ候補と
連続プレイ例を各sourceから示す。定量的な全体集計には同じ規則を全splitへ適用する。

`pose_windows.py`はこのmanifestだけを使う新モデル用dataset。後から承認されたclipを
足さず、crop/resizeを行わずにMDD・pose・observed座標を同じ32実frameで返す。
人物軸だけをbatch内paddingし、時間の反復や欠損位置の補完はしない。
