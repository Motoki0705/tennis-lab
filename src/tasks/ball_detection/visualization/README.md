# Ball Detection Web UI 利用ガイド

原画像にボールのGT・予測位置とheatmapを重ね、クリップを再生して確認します。
[共通の実行前確認](../../base/visualization/README.md#実行前確認)を済ませ、コードのあるリポジトリまたはworktreeの直下で実行してください。各ブロックは単独でコピーできます。

## データセット閲覧

```bash
ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
./scripts/run_in_repo_venv.sh python -m src.tasks.ball_detection.scripts.review_dataset \
  --project-root "$ROOT" --data-root "$ROOT/data" \
  --port 8776
```

[閲覧UIを開く](http://127.0.0.1:8776)。左でBall storeのversionとclipを選びます。checkpoint・GPUは不要です。

## プレイ区間の候補

clipを選ぶと、画像の下に「プレイ／除外候補」「教師窓の被覆」「ボール存在の証拠」の
タイムラインを表示します。クリック、矢印キー、前後の区間境界ボタンで該当frameへ移動でき、
再生・シークにも現在位置と状態が追従します。frame表示は0始まり、時刻は先頭からの実PTS秒です。
推定条件と件数を併記します。除外候補は非プレイの確定ラベルではなく、
短いプレイや位置教師不足も含み得ます。これは表示機能で、区間の編集・承認・保存は行いません。
区間アルゴリズムと学習対象の固定方法は[プレイ区間の契約](../data/PLAY_INTERVALS.md)を参照。

今回のpose承認済みsubsetを確認する場合:

```bash
ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
./scripts/run_in_repo_venv.sh python -m src.tasks.ball_detection.scripts.review_dataset \
  --project-root "$ROOT" --data-root "$ROOT/data" \
  --play-poses "$ROOT/data/ball_detection/ball-mix-v2-player-pose-v1" \
  --port 8776
```

左の **Pose承認済み** datasetを選びます。pose manifestに紐づいた固定ball snapshotを使い、
liveの`ball-mix-v2`とは別に表示します。`--play-poses`はdata root内、参照snapshotはproject root内に
配置してください。通常のBall storeを選んだ場合は全clipの注釈だけから計算し、pose承認条件は適用しません。

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

## パスと注意点

- `--data-root`は**`data`**です。実体は`data/ball_detection/<version>`です。
- 推論の窓長はモデル・clip長に制約されます。
- 未レビュー・未確定・推定ラベルのフレームは採点対象外です。一致検出がない平均距離はN/Aで、誤差0ではありません。
- 別の保存先を使う場合は各root引数を実際の絶対パスに置き換えます。
- ポートが使用中なら`--port`を空き番号へ変更し、その番号のURLを開きます。終了はCtrl+Cです。

[データ・checkpointの契約](../README.md#データセットレビュー--推論ui) / [共通画面・HTTP API](../../base/visualization/detection/README.md)
