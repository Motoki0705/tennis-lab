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

統計画面では、数値表の前に次のグラフを表示します。

- 球注釈の構成：source／対象範囲を切り替える100%積み上げ。分母は球注釈数です。
- 欠損・補間・速度の分布：P5–中央値–P95と平均。全サンプルとclip中央値の分布を切り替えます。
- 位置・動き：位置割合、セル別平均速度、累積移動量と平均変位の矢印を切り替えます。観測なしは斜線で、実測ゼロと区別します。
- stride比較：窓数と一意frame被覆率を別パネルで表示。分母と縦軸の拡大を選べます。
- pose：関節別の速度・相対速度・一瞬の突出をP5–P95で比較します。
- clip散布図：速度P95と欠損長P95から確認対象を探し、クリック／Enterで詳細へ移動します。

ホバーまたはキーボードフォーカスで元の数値・有効件数を表示します。「SVG保存」で軸・単位・凡例を含むベクター図を保存できます。
窓内位置の教師／欠損率のグラフは「32frame窓の位置別グラフ・詳細数値」にあります。
狭い画面では図の領域内を横スクロールできます。分位点からヒストグラムや四分位範囲を推測して描く処理はありません。
