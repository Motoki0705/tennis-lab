# Ball Detection 可視化ガイド

## 学習段階の発表素材

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.visualize_learning_stages input_panel=schematic
```

CPUだけで、事前学習 → 事後学習 → 継続学習の60秒のMP4・GIFと、
`intro / phase1 / handoff12 / phase2 / phase3` の静止画を出力します。
設定の正本は [`configs/visualize_learning_stages.yaml`](../configs/visualize_learning_stages.yaml)。
保存先は `paths.output_root` 配下の
`ball_detection/visualize/learning_stages/<tennis_run_id>/presentation/` です。
同じrunの `hydra/` にCLIログを保存し、可視化ファイルの保存先とは分けます。
既存ディレクトリには上書きしません。worktreeから元repoへ素材を保存するときは、
`paths.output_root=<元repoの絶対パス>/outputs` を明示してください。

`input_panel` は必須です。`schematic` はRGB/MDDの抽象アイコン、`real_clip` は
`input_clip={store_dir: <data-root相対store>, clip_id: <ID>, frame: <窓の開始frame>}` を指定します。
別treeのstoreには `paths.data_root=<dataの絶対パス>` も明示します。
実clipは32frameを読み、保存済み入力契約の `RGBToMDD.from_contract` をCPUで適用し、
窓内frame 1 / 16 / 31のRGBとMDD（G＝明るくなった画素、R＝暗くなった画素）を表示します。
入力不備を模式図へ読み替えず、checkpointの重みも開きません。

評価値・設定・checkpoint記録は `runs.pretraining / runs.posttraining` のrepo相対bundle、
loss・学習率・凍結状態は `training_summary` の無加工の列抽出から読みます。
抽出の由来・検証・再生成は [実測系列asset](../../../../assets/ball_detection/learning_stages/README.md) を参照してください。
評価schema、選択checkpointのSHA-256とstep、固定manifest、凍結状態と学習率を検証します。
横軸は事後学習に事前学習の総更新数を加え、48kの選択checkpointからの引き継ぎを点線で示します。
学習lossは段階ごとに過去20点の移動平均を独立して正規化し、別々の軸で表示します。
継続学習には実測値を作らず、破線と「構想」バッジで循環のみを描きます。

`manifest.json` に入力hash・checkpoint hash・git commit・解決済みconfig・最終Common/Full値と
由来（Phase 1/2＝measured、入力模式図／アーキテクチャ＝schematic、Phase 3＝concept）を記録します。
生成物はユーザー確認前の候補です。現在の継続学習の実装状況は
[注釈・継続学習](../../base/annotation/README.md)を参照し、この構想動画を実証結果として扱いません。

## Web UI

保存RGB画像にボール注釈・予測位置とheatmapを重ね、クリップを再生して確認します。
[共通の実行前確認](../../base/visualization/README.md#実行前確認)を済ませ、コードのあるリポジトリまたはworktreeの直下で実行してください。各ブロックは単独でコピーできます。

## データセット閲覧

```bash
ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
./scripts/run_in_repo_venv.sh python -m src.tasks.ball_detection.scripts.review --mode dataset \
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
区間アルゴリズムと学習対象の固定方法は[プレイ区間の契約](../data/windows/README.md)を参照。

今回のpose承認済みsubsetを確認する場合:

```bash
ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
./scripts/run_in_repo_venv.sh python -m src.tasks.ball_detection.scripts.review --mode dataset \
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
./scripts/run_in_repo_venv.sh python -m src.tasks.ball_detection.scripts.review --mode inference \
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

## 発表用の単一パネルGIF・MP4

既存の `scripts.visualize` で `visualization.layout=overlay` を指定します。
保存RGBを `visualization.overlay.output_width`（既定960）へ縦横比を保ってresizeし、
推論位置だけを重ねます。文字・header・補間点は描かず、不可視frameでは元画像だけを表示します。
既定のマーカーは案Aの細い輪＋中心点＋薄いglowです。モデル入力の解像度は変更しません。

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.visualize \
  visualization.layout=overlay visualization.overlay.output_width=960 \
  visualization.clip_id=tracknet/game1/Clip1 visualization.max_frames=90 \
  visualization.fps=15 \
  'run.output_dir=ball_detection/visualize/presentation/ball_2d/${tennis_run_id:}' \
  'visualization.save=${run.output_dir}/ball_2d.gif'
```

storeの途中の区間は `visualization.start_frame`（0始まり、既定0）と
`visualization.max_frames`（その位置からの最大frame数）で指定します。たとえば
`visualization.clip_id=meiji/video_000/clip_000/cam1 visualization.start_frame=200 visualization.max_frames=100`
はframe 200〜299です。負の開始位置・clip末尾以降・モデル窓より短い区間は拒否します。
モデル入力・MDDの前処理はclip全体と同じ経路です。

### 元動画の速度とマーカー案の比較

`visualization.fps=null` はstoreに記録された元動画のfpsを使う指定です（overlayのみ）。
`visualization.overlay.frame_stride` は**推論後の描画だけ**を間引き、GIF・MP4のfpsは
元動画のfps / strideになります。モデルは選択区間の全frameを、元の順序と時間文脈で推論します。
たとえば59.94006fpsの1010frameをstride=2にすると、505frame・29.97003fpsで
約16.85秒の1x再生になります。端数の区間では最後の表示もstride分だけ保持します。
数値の `visualization.fps` は従来どおり指定した表示fpsを使います。

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.visualize \
  visualization.layout=overlay visualization.clip_id=meiji/video_000/clip_000/cam2 \
  visualization.start_frame=0 visualization.max_frames=null visualization.fps=null \
  visualization.overlay.output_width=960 visualization.overlay.frame_stride=2 \
  'visualization.overlay.styles=[ring,trail]' \
  'run.output_dir=ball_detection/visualize/presentation/ball_2d/${tennis_run_id:}' \
  'visualization.save=${run.output_dir}/ball_2d.gif'
```

`overlay.styles` は `point`（従来の輪）、`ring`（案A、既定）、`trail`（案B）の重複しないリストです。
複数指定すると、**1回の推論を共有**し、保存先の `ring/ball_2d.gif`、`trail/ball_2d.gif` と
同名の `.mp4`、各 `manifest.json` を出力します。1種類だけの場合は指定したGIFパスと
同名の `.mp4` へ保存します。overlayは両形式を必ず出力し、スライドではMP4を使います。

案Aは `overlay.ring` で、半径9・線幅2・中心点半径2・glow半径3・opacity 0.12を明示します。
案Bは `overlay.trail` で、点半径4・線幅2・長さ8（元動画のframe数）・最大opacity 0.55を明示します。
マーカー寸法は出力幅960を基準に比例させ、色は共通の `draw.pred_color_rgb` を使います。
軌跡は古いほど薄く、**予測のない元frameで切ります**。表示を間引いても欠測をまたいで結びません。
不可視frameではマーカーも軌跡も描かず、文字・補間点は追加しません。
manifestには全推論frame、表示frame、元fps、描画stride、出力fpsとstyleを記録します。

GPU推論は[共有キュー](../../../../.agents/skills/training-queue/SKILL.md)へ登録して実行してください。
worktreeでは実データの `paths.data_root` とcheckpointのrootを絶対パスで指定します。
checkpointは `visualization.checkpoint` の `role`（`artifact` / `checkpoint`）と `path` で
保存先を明示します。既定の参照は従来どおりartifact root配下です。

overlayではGIFの隣に `manifest.json` を保存します。入力のstore tables・clip payload・checkpointの
SHA-256、コードのcommitと未コミット変更の有無、解決済みconfig、frame名・frame範囲（stopは含まない）、
映像と予測の由来区分 `measured` を記録します。予測点は実データへの推論結果であり、GTではありません。
`outputs.gif`・`outputs.mp4` に両ファイルのパスとSHA-256を記録します。
MP4は共通の `src.utils.video.writer.VideoWriter` で、GIFと同じ描画frame・fpsを
H.264へエンコードします（writer既定のCRF 17）。GIFの表示時間は共通writerの10ms単位に量子化されます。
MP4のyuv420p形式には偶数の幅・高さが必要で、不正な寸法はwriterが拒否します。

既定の `visualization.layout=grid_2x2` は従来の4パネル・文字・モデル入力解像度・12fpsを維持します。
4パネルの装飾設定は `visualization.layout.<項目>` から `visualization.grid.<項目>` へ移しました。
不明なlayout名は実行前に拒否します。

## モデル候補とCodex確定注釈の発表用比較

`presentation/annotation_correction.py` は、旧 `campaign_ball_candidates.v1` の候補と
`tennis_chat_ball_annotation.v1` の確定注釈を、同じframeに重ねて左右で比較します。
モデル候補は探索の手がかりであり、右側はCodexが確認した注釈です。
新しい推論は実行せず、CPUだけで既存成果物を読みます。
旧形式の契約は [migration reader](../../annotation/migration/README.md) を使います。

採用素材は `csu_MWYHS8k__3b250a3c0ef22d47__f000002998-000003776` の
frame `217:517`（0始まり・終了を含まない300frame、5.005秒）です。
元MP4は削除済みのため、採用された `ball-mix-v2` の保存RGBを
`frames.source=store` で明示選択します。configの既定値はこの素材・区間と1920×1080です。
以下をworktree直下から実行すると再生成できます。入力rootは共有データの**絶対パス**で、
別環境では3つのrootを既存データに合わせて変更してください。
出力は現在のworktreeの `outputs/ball_detection/visualize/annotation_correction/<run-id>/presentation/`
です（`paths.output_root` で変更可能）。入力への書き込みはありません。

```bash
ANNOTATION_ROOT=/home/kamimura/projects/tennis-lab/outputs/chat_annotation
CAMPAIGN_DIR="$ANNOTATION_ROOT/local_agent"
DATA_ROOT=/home/kamimura/projects/tennis-lab/data

.venv/bin/python -m src.tasks.ball_detection.scripts.visualize_annotation_correction \
  mode=render frames.source=store frames.store_dir=ball_detection/ball-mix-v2 \
  clip_id=csu_MWYHS8k__3b250a3c0ef22d47__f000002998-000003776 frames.range=217:517 \
  resolution.width=1920 resolution.height=1080 \
  paths.annotation_root="$ANNOTATION_ROOT" paths.campaign_dir="$CAMPAIGN_DIR" \
  paths.data_root="$DATA_ROOT"
```

1920×1080では
左右944×531px・間隔32px、その下に36pxの全clip分類帯と現在frameの白線を描きます。
文字は描きません。異なる入力アスペクト比はパネル内に余白を入れて保ちます。
読み込み時にschema・`proposals_only=true`・候補の降順・注釈の寸法と全frame対応を検証し、
renderでは入力のSHA-256・PTS・表示時間をmanifestと照合します。
元FPSの固定レートで全frameを再生し、可変frame間隔の動画は拒否します。

`frames.store_dir` は明示したDATA rootからの相対パスです。storeの
`chat_annotation/<clip_id>` を使い、`media_sha256` が候補・元clip manifestの動画hashと
一致すること、全frame数・元解像度・FPS・各frameのPTSが一致することを検証します。
候補と確定注釈の座標は元画像pxのまま判定し、保存RGBの縮小率を描画に反映します。
manifestの `frame_input` に入力元、store・metadata・index・画像payloadのpath/hash、
保存／元解像度、座標倍率、JPEG品質を記録します。元動画hashは元メディアの同一性を表し、
保存JPEGの画素を表すhashとは区別します。

元MP4がある素材では、同じrenderコマンドに `frames.source=video frames.store_dir=null` と
clip・区間を明示して動画から復号できます。入力元の自動切り替えはありません。

別素材を探す場合は採用値を解除して順位付けします。rankはJSONだけを検証し、
RGBを読みません（`frames.source=video` はrender用store設定を解除するための指定です）。

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.visualize_annotation_correction \
  mode=rank clip_id=null frames.range=null frames.source=video frames.store_dir=null \
  paths.annotation_root="$ANNOTATION_ROOT" paths.campaign_dir="$CAMPAIGN_DIR"
```

`ranking.json` はcandidate cacheとprocessed注釈が揃う全clipのclass別frame数と入力hashを記録します。
4種類の変更がすべてあるclipを先にし、変更frame比率の降順で並べます。
候補の順位を並べ替えたり、欠落frameを補ったりしません。
別のclip・区間の採用は、ランキングと短いプレビューを見て利用者が決めます。

帯は未確認＝灰、順位変更＝黄、位置指定＝橙、追加＝緑、除去＝赤です。
一致・双方なしは無色で、距離は `radius_px=8` を1920px幅の基準から元画像幅へ換算します。
球が複数ある場合はtrackごとに判定し、除去→位置指定→追加→順位変更→一致の順で優先します。
座標のない球も除去／双方なしの判定に含みます。未確認frameの右側には何も重ねません。
左は最上位を白円、2・3位をscoreに比例した濃さの点で示します。
右は可視＝緑の塗り円、遮蔽＝破線円、補間＝中空円です。
変更classの連続区間の開始から0.3秒で矢印を伸ばし、枠を0.4秒点灯します。

`playback.hold_on_change_frames=0` が既定です。正数を明示した場合だけ、変更のある各frameを
指定数だけ追加複製します。manifestに元frameとの全対応と複製数を残します。
MP4・GIF・class別の先頭静止画・manifestを共通exporterで書き出し、既存の出力先は拒否します。
manifestは入力・checkpointのhash、git commit、解決済みconfigと由来を記録します。
左右の結果と帯は実データ (`measured`)、矢印・枠は説明用の図示 (`schematic`) です。

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
