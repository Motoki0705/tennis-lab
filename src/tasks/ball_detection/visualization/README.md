# Ball Detection Web UI 利用ガイド

保存RGB画像にボール注釈・予測位置とheatmapを重ね、クリップを再生して確認します。
[共通の実行前確認](../../base/visualization/README.md#実行前確認)を済ませ、コードのあるリポジトリまたはworktreeの直下で実行してください。各ブロックは単独でコピーできます。

## データセット閲覧

```bash
ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
./scripts/run_in_repo_venv.sh python -m src.tasks.ball_detection.scripts.review_dataset \
  --project-root "$ROOT" --data-root "$ROOT/data" \
  --port 8776
```

[閲覧UIを開く](http://127.0.0.1:8776)。左でBall storeのversionとクリップを選びます。checkpoint・GPUは不要です。

1. 「体系・内訳を見る」で現物のsource・split・frame数とinstance数、入力と教師、派生poseの対応範囲を確認します。
2. Source・Split・注釈状態でクリップを絞ります。注釈状態を選ぶと、該当状態の先頭frameを開きます。
3. 右の「現在のフレーム」で採点対象か参考ラベルかを読みます。観測は緑、補間は黄、遮蔽推定は紫の小点です。座標がない位置不明・画面外も一覧に残ります。
4. 下の「注釈状態へ移動」で前後の該当frameを確認します。再生は選んだ状態で間引かず、全frameを表示します。

体系の説明は[データセット体系](../data/README.md)を参照してください。

## Player pose・tracking

右の「Player pose / Tracking」で、`data/ball_detection`配下のposeデータセットを選びます。
対応するデータセットがあれば自動で選択され、採用済みの選手だけを標準表示します。
「生成結果」へ切り替えると、未採用・保留を含む全人物のraw trackingを確認できます。
切り替えても現在のフレーム位置は維持します。

姿勢（COCO-17）、人物枠、ID、軌跡は個別にON/OFFできます。色は表示中のIDに対応し、
採用結果では匿名選手ID（P）と対応するraw ID（R）を小さく併記します。IDはクリップ内の対応です。
姿勢はモデルによる推定で、レビューは人物の役割・ID対応を採用するものです。
軌跡は人物枠の下辺中央を最大20コマ表示し、欠損や区間境界で途切れます。
遠くの小さい選手は細い骨格線を優先し、ズームすると顔・関節点も表示します。
ボールのGT・予測は小さな塗りつぶしの点で表示します。

一覧のPlayer状態で、採用済み・要確認・レビュー待ち・未生成・対象外・生成エラーを
絞り込めます。「この結果は未提供」と「0人観測」は区別します。追加クリップにposeが
まだなければ未生成として表示します。更新された結果はカタログ更新ボタンで読み込みます。
閲覧は保存済み結果の読み取りだけで、playerの自動処理を起動・再開しません。

## 推論

```bash
ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
./scripts/run_in_repo_venv.sh python -m src.tasks.ball_detection.scripts.inference_ui \
  --project-root "$ROOT" --data-root "$ROOT/data" \
  --outputs-root "$ROOT/outputs/ball_detection" \
  --checkpoints-root "$ROOT/ckpt/ball_detection" \
  --port 8777
```

[推論UIを開く](http://127.0.0.1:8777)。閲覧と同時に使う場合は別ターミナルで起動します。

1. 左でcheckpointを検索・選択します。モデルの最小窓に足りないクリップは候補から外れます。
2. dataset・クリップを選び、右で開始フレーム・フレーム数・検出しきい値を指定します。
3. DeviceをCUDAにして推論を実行します。CPUで試す場合はCPUを明示選択します。GPU要求は[共有キュー](../../base/visualization/README.md#web-uiのgpu実行)で実行されます。
4. 中央で再生・シークし、GTと予測位置を比較します。右でprobabilityレイヤーを選ぶとheatmapも確認できます。

## 表示操作

ドラッグで画像を移動、ホイールでズーム、フィットで全体表示へ戻します。下部で再生・一時停止・シーク・速度を調整し、ダウンロードボタンで表示PNGを保存できます。

再生速度は「毎秒表示するコマ数」です。たとえば元動画が60fpsでも、25fps指定では
全フレームを毎秒25コマずつ進めます。先読みが追いつかない場合はフレームを飛ばさず、
「読み込み待ち」を表示します。実測fpsは直近約2秒の表示更新から算出します。
ブラウザーのタブを隠すと一時停止します。

## パスと注意点

- `--data-root`は**`data`**です。実体は`data/ball_detection/<version>`です。
- 推論窓の長さはモデル・クリップ長に制約されます。
- 未レビュー・未確定・推定ラベルのフレームは採点対象外です。一致検出がない平均距離はN/Aで、誤差0ではありません。
- 別の保存先を使う場合は各root引数を実際の絶対パスに置き換えます。
- ポートが使用中なら`--port`を空き番号へ変更し、その番号のURLを開きます。終了はCtrl+Cです。

[データ・checkpointの契約](../README.md#データセットレビュー--推論ui) / [共通画面・HTTP API](../../base/visualization/detection/README.md)
