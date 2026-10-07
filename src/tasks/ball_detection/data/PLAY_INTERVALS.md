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

レビューは[データセットWebUI](../visualization/README.md#プレイ区間の候補)で行う。
選択したclipの注釈からCPUでその場で計算し、画像・GTと同じframe列に区間を表示する。
ファイル出力・モデル推論・学習は行わない。通常storeの候補とpose承認済みsubsetは
別のdatasetとして表示し、後者はpose manifestが指定したball snapshotを検証して読む。
カタログ更新で承認状況を再読込するため、UI表示自体は学習対象の凍結や承認ではない。

レビュー後に学習対象を固定するAPIは
`play_manifest.build_play_manifest(pose_directory, PlayIntervalConfig(...))`。
返り値をJSONとして実験出力先に保存すると、approved subset、ball snapshot、
pose artifact、JPEG shardのhash、学習窓と全除外範囲を固定できる。
この保存処理はWebUIの閲覧とは分離し、生成物をソースリポジトリへ追加しない。

`pose_windows.py`は固定したmanifestだけを使う新モデル用dataset。後から承認されたclipを
足さず、crop/resizeを行わずにMDD・pose・observed座標を同じ32実frameで返す。
人物軸だけをbatch内paddingし、時間の反復や欠損位置の補完はしない。
