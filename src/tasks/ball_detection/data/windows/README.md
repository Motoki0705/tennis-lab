# プレイ区間・32frame学習窓のレビュー

`intervals.py`は注釈からプレイ区間を提案する純粋関数。
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

レビューは[データセットWebUI](../../visualization/README.md#プレイ区間の候補)で行う。
選択したclipの注釈からCPUでその場で計算し、画像・GTと同じframe列に区間を表示する。
ファイル出力・モデル推論・学習は行わない。通常storeの候補とpose承認済みsubsetは
別のdatasetとして表示し、後者はpose manifestが指定したball snapshotを検証して読む。
カタログ更新で承認状況を再読込するため、UI表示自体は学習対象の凍結や承認ではない。

レビュー後に学習対象を固定するAPIは
`pose_manifest.build_play_manifest(pose_directory, PlayIntervalConfig(...))`。
返り値をJSONとして実験出力先に保存すると、approved subset、ball snapshot、
pose artifact、JPEG shardのhash、学習窓と全除外範囲を固定できる。
この保存処理はWebUIの閲覧とは分離し、生成物をソースリポジトリへ追加しない。

## FPSと32frame窓

上のAPIと既存WebUIはnative-FPSの候補を扱う。
`temporal.py`はプレイ区間内で元FPS/1/2/1/4（`frame_step=1/2/4`）の32枚を選び、
選んだframeの証拠率・observed教師数を改めて判定する。
窓のframe列は`start + frame_step * arange(32)`、開始strideの単位は**sampled frame**。
`window_stride=16`なら元frameでの開始間隔は16/32/64枚となり、区間末尾を覆う窓も追加する。
プレイ区間はnative PTSから決め、窓はreference-only frameと大きいPTS gapを跨がない。
時刻は選んだframeの実PTS秒を使い、一定FPSの時刻を捏造しない。

`dataset.py`は固定manifestからRGB・任意のpose・observed座標と教師maskを同じ32枚で返す。
RGBを間引いてからモデル内でMDDを再計算する。native-FPSのMDDを間引かない。
5ch構成でもdatasetは同じRGBを渡し、拡張後にモデルがRGBとMDDを結合する。
入力契約は [models/mdd](../../models/mdd/README.md) を参照。
crop/resize、時間の反復、欠損位置の補完はしない。
座標lossは`training/coordinates/objective.py`が担当する。

## 学習manifestの凍結

`manifest.py`は`mdd_pose`をpose承認済みclipから、`mdd_only`をposeの有無によらず全clipから選ぶ。
`mdd_only`にはclip全体の存在率による足切りを設けず、プレイ区間と窓ごとの条件を使う。
Common評価はpose参照とGT・split・timeline・RGBのidentityが一致するsubset、Full評価は全validation。

レビュー後、次の公開入口で新しい絶対pathへ凍結する（学習は開始しない）:

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.prepare_data \
  --ball-store /absolute/ball-store --poses /absolute/approved-poses \
  --model-config /absolute/configs/model/mdd_pose.yaml \
  --output /absolute/new-prepared-data --frame-steps 1 2 4 --window-stride 16
```

`preparation.py`はindex/metadataとpose artifactをコピーし、RGB shardを`ball_snapshot/`へhardlinkする。
`mdd_pose_windows.json`・`mdd_only_windows.json`に窓・除外clip・入力hash・評価subsetを保存し、
`models/`に比較設定を出力する。各split/FPSの窓とCommon評価窓を検証した後に
`experiments.json`を公開し、`PREPARATION_INCOMPLETE`を取り除く。完了したplanのmanifest/model pathを
[学習入口](../../training/README.md)へ渡す。凍結出力はソースリポジトリへ追加しない。

## snapshotを共有するrunの運用

同じ元storeのshardを共有するsnapshotを学習・評価runが読んでいる間は、
`prepare_data`によるsnapshotの追加と、既存snapshotの削除を行わない。
hardlinkの作成・削除は共有inodeのctimeを変えるため、RGB内容が同じでも、
検証済みdatasetの次回読込や`FileRangeReader`の範囲読込は変更を検出して停止する。
必要なsnapshotをrun開始前にまとめて準備し、削除は共有する全runの終了後に行う。
この検査はmtimeを戻した内容変更も検出するため、ctimeの照合を省略しない。
