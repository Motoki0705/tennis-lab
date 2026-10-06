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

clipを選ぶと、画像の下に「プレイ／除外候補」「教師窓の被覆」「位置注釈」「選択用の証拠」の
全体タイムラインを表示します。位置注釈は画像上に表示できる座標に対応し、青は実測、
斜線は補間・遮蔽位置推定です。参照用frameの位置注釈も表示します。
選択用の証拠は区間推定に使う条件であり、参照用frameは含めず、位置未確定の球は含めます。
現在frameの注釈種別と証拠から除外した理由を画像と同期して表示します。

全体図では短い欠損が潰れるため、**16／32／64frameの拡大欄**で1frameずつ確認できます。
1frameは最低16pxで、画面が狭ければ横スクロールします。セルのクリックや矢印キーで移動し、
セルのツールチップでframe番号・時刻・注釈状態を確認できます。
「前／次の注釈変化」は位置注釈の有無・実測／推定の変化、「区間境界」はプレイ候補の境界へ移動します。
再生・シークにも現在位置と拡大欄が追従します。frame表示は0始まり、時刻は先頭からの実PTS秒です。
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

## データセット統計

BallのDataset Reviewでdatasetを選び、画面上部の「データセット統計」を開きます。
比較stride、採用範囲のstride、位置分布の分割数、pose候補閾値を確認し、「統計を計算」を押します。
CPUで進捗を表示しながら計算します。poseも調べる場合は`--play-poses`で起動し、Pose承認済みdatasetを選んでください。

source・split・対象範囲で絞り込むと、位置分布、領域間遷移、注釈構成、stride比較を確認できます。
「分布統計」で全サンプルの分布とclip間の分布を切り替え、指標名で検索できます。
clip一覧は速度・欠損・補間・pose候補で並べ替え可能です。clip詳細には原注釈のissues/notesと確認候補を表示し、
frameボタンや軌跡の点をクリックすると画像レビューへ移動します。

設定変更後は再計算してください。同時に計算できるのは1件で、最新の結果だけ保持します。
取得できない情報は理由と未定義値で表示します。指標の定義・分母・制約は
[統計仕様](../dataset_statistics/README.md)を参照してください。
