# 実RGBからSLCS学習データを作る

Meijiの同期3カメラと放送映像を、検出観測・品質重み付き3D擬似ラベル・DINOv3特徴へ変換する。
出力階層と実験名の付け方は[タスク出力規約](../../tasks/OUTPUTS.md)を正本とする。
モデルの学習結果・比較数値は `knowledge/nodes/run-slcs-*` に記録する。

## 1コマンドで生成

repoまたは専用worktreeのrootから実行する。

```bash
bash scripts/datasets/build_real_rgb.sh all
```

共有training queueへ投入し、broadcastの保存済み2Dボールを取り込み、両データの観測・
教師推論・幾何補正・RGB特徴を生成し、固定split付きのSLCSデータへ統合する。
queue状態は元repoの `.training_queue/` で確認する。worktreeごとのqueueは作らない。
AI実行時は `TENNIS_QUEUE_PROVIDER` と `TENNIS_QUEUE_SESSION` を明示する。

生成の正本設定は [`build_slcs_dataset.yaml`](../configs/build_slcs_dataset.yaml)、
[`build_broadcast_slcs_dataset.yaml`](../configs/build_broadcast_slcs_dataset.yaml)、
[`assemble_slcs_dataset.yaml`](../configs/assemble_slcs_dataset.yaml)。
保存先・除外理由・固定カメラの校正クリップ・採用閾値・教師重みはここで指定する。
既存の入力データへアノテーションを上書きしない。

1クリップだけを独立したデータ版として生成する例:

```bash
bash scripts/datasets/build_real_rgb.sh meiji \
  'dataset_clip_ids=[video_000/clip_000]' \
  dataset_output_directory=slcs/meiji_one_clip_v1 \
  output_dir=tennis_scene/generate/meiji_one_clip_v1/s42-001
```

`dataset_clip_ids` はデータ版の収録対象を固定する。`clip_ids` は、その収録対象のうち
今回処理する部分だけを選ぶ再開用の指定。除外は `excluded_clips` に理由付きで残す。
別の入力を処理する場合は `dataset_directory`、`ball_source`、座標モード、校正設定を
新しいrecipeへ明示する。入力manifestには同期・frame数・サイズ・camera IDが必要。
本経路は固定カメラ、2人のsingles、クリップ内のend change無しを前提とする。

## 必要な入力と重み

Pythonはrepoの `.venv/bin/python`。DINO detectorのCUDA拡張は実行環境のPyTorch/CUDAで
ビルドされている必要がある。外部モデルの実体は `paths.external_asset_root` で指定する。
wrapperはworktreeから元repoの `third_party` を参照する。

Courtには設定で選定した **`outputs/court_detection/` 内のcheckpoint** を使う。
Meijiではball cropで初期Courtを推定し、その14点を含む範囲へ
`court.crop_refinement_padding_px`（20px）を加えたcropで再推論する。
同じframe・checkpoint・fit閾値を使い、2回目のfit失敗は生成失敗とする。
full画像を使うcameraは1回の推論を維持する。初回・再推論のraw点、ROI、H、
診断を別々に保存する。この設定を省略したrecipeは従来の1回推論を維持する。
Courtの変更は人物選択と3D教師に影響するため、Meijiは新しいv9と観測rootへ分離する。
Meijiのボールは各clipの `outsource/cam*_annotations.json` のみを取り込み、
画像動画のSHA-256・frame数・statusを検証する。ボール検出器は実行しない。
放送側は承認済みの保存 `scene.npz` から2Dボールだけを取り出し、孤立したUV spikeを除く。
保存された古い人物・Court・3D値は新しい教師へ引き継がない。

PLCS/BLCSの配置先はbuild設定の `plcs_checkpoint` / `blcs_checkpoint`。
60epoch学習から**validation位置誤差最小**の重みを選び、`ckpt/` にコピーする。
隣の `*.metadata.json` に元のrun、epoch、SHA-256、選定指標を記録する。
test値を重み選択に使わない。DINOv3とViTPoseの配布重みも設定されたrootへ配置する。
不足する重みがあれば生成開始時にエラーとなる。

配布重みを再取得するときは、取得元のrevisionを固定し、別ファイルへdownloadしてサイズ・
期待SHA・モデル構造を確認する。元ファイルをバックアップしてからatomicに切り替え、
取得元と検証結果をknowledgeのrunへ残す。既存の固定SHAと一致する場合は同じモデルなので、
pinやdataset版を変更せず、入力・設定が一致するcacheを再利用できる。
異なる重みを採用する場合は新しいpinとデータ版が必要。再取得の成功と、過去の読取不一致の
原因解決は区別し、実際の推論と生成した教師の品質も確認する。

Meijiの `checkpoint_sha256` は採用時に確認した6モデルの期待SHAを固定する。
各stageの必須checkpointを開始前に照合し、結果をrun内の `checkpoint_verification.json` に残す。
この設定を指定するrecipeは6種類の役割と64桁の小文字hexをすべて明示する。
期待値は採用記録を固定するためのもので、配布元による署名を意味しない。
人物観測ではcameraごとにもDINO/ViTPoseのSHAを固定値へ照合し、推論前後の
checkpointファイルの一致と、raw検出・人物poseのreceipt間のDINO SHA一致を確認する。
既存cacheの利用時と `stage=infer` による観測の消費時にもreceiptを照合する。
不一致のreceiptを期待値へ書き換えず、元の実測値を保持する。
ファイル照合には共通 `src/utils/checksum.py` を使う。読取中の変更、short read、二実装間の
不一致は生成を停止し、clip処理中なら当該clipの失敗も保存する。自動retryはしない。

`checkpoint_warning_roles` は既定では空で、期待SHAや推論前後・receipt間の不一致も停止する。
Meijiのrecipeだけはユーザーの続行指示に基づき `[dino, vitpose]` を明示する。
この2役割のdigest差は `checkpoint_warnings.jsonl` へrole・path・比較箇所・実測値・期待値を
記録し、宣言した固定pinをモデル識別としてcache照合を継続する。新receiptでは
`declared_checkpoint_sha256` と実測digestを分け、旧receiptの宣言値を仮定する場合も警告する。
異なる宣言pinや他のidentity項目、動画・観測配列・教師の品質条件はこの例外に含めない。
これは指定checkpointの内容認証を省略できる運用であり、同一bytesの読み込みや原因解決を
保証するものではない。品質レポートにも実測digest・宣言値・警告・認証限界を分けて残す。
監査は生成runへ追記せず、監査時の警告を新レポート内に保存する。
方針変更後の生成はv9の `s42-002` としてrecipeを分離する。v9はその時点で3D教師未公開であり、
固定pin・Court・媒体・特徴生成条件が同じ既存Court/RGBを保持して再開する。

## 品質判定と教師の意味

`court.py` が複数frameのCourt観測から静的homographyを推定する。
Meijiは同じ収録・同じcamera/cropであることを確認して校正を共有する。
`people.py` はCourt半面ごとの人物検出を関連付け、ViTPoseで2D関節を得る。
人物観測cacheのP0/P1は各視点のnear/farであり、反対側から撮るカメラでは同じ番号が
別選手を指す。教師入力は `associate_people` が `view_half_turns` に従って共通の
reference courtの人物軸へ並べ替える。視点間の支持率はこの並べ替え後に集計し、
raw観測の画像とcanonical軸の教師画像を区別する。
Meijiの `people.selection_policy=temporal_continuity` は最初の最大boxを起点に、
直前に選択した実検出のboxと、その移動から予測したboxの両方に対するIoU・中心距離で同じ選手を追う。
予測は直近5区間の速度中央値を使い、`people.association` に指定したframe数と移動量で
制限する。中心距離は直前boxの対角長で正規化し、同設定の固定閾値を使う。候補の大きさで途中から選び直さず、
欠損後も閾値を広げたり別人で初期化したりしない。初期選択の誤りや人物同士の重なりは
この規則だけでは判別できないため、映像による対応確認も行う。broadcastは検証済みの
従来設定（省略時の `largest`）を使い、選択方式・閾値の変更は別の観測cache版へ保存する。
検出率が基準を満たさない場合は停止する。`people.long_gap_policy=error`（省略時も同じ）では
最長欠損の超過も停止する。Meijiの明示設定 `mask` では長い欠損と非観測のクリップ端を
`pose_supported_mask` で除き、関節confidenceを0として残りのカメラから教師を作る。
検出フレームを示す `observed_masks` は別に保持し、短い内部欠損のbox補間と区別する。
Meijiは選手のサイドライン外への移動を含む横幅6.5m、単眼broadcastは5.8mで選択する。
ViTPoseの精度モードはconfigに記録し、DINO detectorは対応するfloat32で実行する。
ViTPoseの生ヒートマップピークは確率ではなく1を超え得るため、再構成時に有限性を検証し、
`[0,1]`へ飽和させた値を関節visibilityとして保存する。変換方式・生値の範囲・飽和件数は
scene metadataへ記録し、観測cacheの生ピークは保持する。SLCSの読込側は範囲違反を拒否する。

`refinement.py` は多視点で観測されたボールと腰中心を三角測量し、再投影残差・速度・
高さ・コート範囲で検査する。`refinement.player_root_view_support` は既定の `hips` で
両hip、Meiji v8の `hips_and_shoulders` で両hipと両shoulderのconfidence ≥ 0.3を
各カメラの参加条件とする。三角測量する点はどちらも両hip中心であり、肩は支持判定だけに使う。
部分的に画面外へ出る選手への対策で、全収録への一般化や独立3D精度は未検証。
速度・再投影・coverage閾値と補間規則は変更しない。broadcast v4は既定の `hips` を維持する。
短い欠損の補間と、モデル予測だけの区間を別source codeで残す。
単眼では選手の足元をCourt平面へ写し、BLCSのボール奥行きは推論値を保つ。
単眼ラベルは低い重み、多視点でも支持のない予測は重み0とし、SLCSの教師から除く。
PLCSのcanonical pose/yawはモデル推論であり、独立した実測ラベルではない。

生成runの `<video_id>/<clip_id>/` に `raw_model_quality.json`、`label_evidence.json`、
`quality.json`、`quality_arrays.npz` を保存する（dataset側ではscene metadataに品質・ラベル重みを保持）。
**再投影の改善は観測との整合性であり、実測3D精度を保証しない。**
同じ観測を補正と評価に使った数値と、別収録のheld-out評価を区別する。
採用判定に失敗したclipは `failures.json` に残り、成功分のcacheを保持してジョブは失敗する。
設定の閾値を自動的に緩めたり、失敗clipを無言で除外したりしない。

## 再開・cache・split

`stage=all` が通常経路。診断用に `court`、`observe`、`infer`、`features` を分離できる。
raw detector、人物観測、Court、3D教師、RGB特徴は入力・設定・checkpoint hashで照合する。
3D教師は構成と2D観測のcontent identityを持つ。異なる教師で作った完成markerを
単にskipすることはない。内容を変える場合は新しいdataset版を指定する。
生成logのrun-idだけを変えても、異なる教師を既存データへ混在させることはできない。

統合は全clipの教師とDINO特徴を検証してから、別ディレクトリへatomicに公開する。
媒体と配列はimmutableなhardlink、manifestと完成markerは新dataset IDに合わせた別ファイル。
`assembly.json` が元データのhashと派生manifestを記録する。`splits.json` の改変も検知する。
同じ収録の別clipをsplit間へ分散させない。broadcastは同じ会場・試合由来のclipをgroup化する。

## 学習と比較

SLCSの実RGB学習profileは [`train_real_rgb.yaml`](../../tasks/slcs/configs/train_real_rgb.yaml)。
探索学習は60epoch。共有queueへ次を投入する（queueへの投入方法はtraining-queue skill参照）。

```bash
.venv/bin/python -m src.tasks.slcs.scripts.train --config-name train_real_rgb \
  run.output_dir=slcs/train/real_rgb_augmented/s42-001
```

入力のみの座標ノイズ、関節・ボール欠損、連続検出欠損、RGB/2Dモダリティ欠損を学習時に付加する。
教師配列とvalidation/testの入力は変更しない。対照は同じprofileに
`data.augmentation.enabled=false` と別run-idを指定する。
評価は `evaluate.input_mode=full|no_rgb|detector_gap|rgb_only` を切り替え、
同一checkpoint・split・教師maskを使う。実RGBを使う効果と検出欠損時の劣化を測る。

```bash
.venv/bin/python -m src.tasks.slcs.scripts.evaluate \
  paths.checkpoint_root=outputs \
  evaluate.checkpoint=slcs/train/real_rgb_augmented/s42-001/logs/version_0/checkpoints/SELECTED.ckpt \
  data.dataset_root=slcs/real_rgb_v1 data.split_file=slcs/real_rgb_v1/splits.json \
  evaluate.input_mode=detector_gap \
  evaluate.output_dir=slcs/evaluate/real_rgb_detector_gap/s42-001
```

各試行の固定コマンド・config・予測・考察はknowledge graphに保存する。
`train_real_rgb_pilot.yaml` は採用済み7クリップで入力欠損施策を比較する先行試験用で、
Meijiのtest収録を含まない。最終評価には全体版と収録単位の固定test splitを使う。
教師の再学習profileは各タスクの `train_broadcast_real_rgb.yaml`、
PLCSの `train_meiji_foot_real_rgb.yaml`、BLCSの `train_meiji_real_rgb.yaml`。
データ分割・subset作成は `scripts/analysis/prepare_plcs_motion_split.py`、
`prepare_plcs_subset.py`、`prepare_blcs_real_dataset.py` のreceiptを伴う処理を使う。
BLCS旧/改善重みの同条件比較には `scripts/analysis/evaluate_blcs_real.py` を使う。

## 全clip品質レポート（CPU）

`build_real_rgb.sh all` はMeiji生成後・統合前にこの検査を実行する。単独で再集計する場合:

```bash
.venv/bin/python -m src.tennis_scene.scripts.report_slcs_dataset_quality
```

入力dataset、期待集合のsource manifest、複数の生成run・観測root、理由付き除外は
[`report_slcs_dataset_quality.yaml`](../configs/report_slcs_dataset_quality.yaml)で明示する。
`tennis_scene/analyze/...` に `quality_report.json` とclip単位の展開済みCSV、実行configを保存する。
欠落・不正データ・教師/DINO provenance混在は成功扱いしない。人物検出・姿勢推定の
checkpoint記録、人物選択policy・設定、Courtのcheckpoint記録・設定もカメラ間・clip間で
照合する。これは記録済みproducerと観測bytesの整合検査で、重み実体の再認証ではない。
欠落を含む進行中のsnapshotは
`allow_incomplete=true` で明示し、その場合も不正データは失敗する。失敗時もレポートは残る。

raw/refined双方に同じ最終SLCSの正重みframe maskを適用し、全軌道の診断と区別する。
速度は隣接両端が支持される区間だけを集計し、欠損を跨がない。これはwindow選択前のframe適格性で、
学習windowの採否・重複によるサンプル頻度は再現しない。全clip集計の平均はサンプル数で重み付けし、
clip別percentileを全体percentileとして平均しない。Court画像homography fitと近似pinhole fitは別欄にする。
全関節の再投影誤差に球/hip三角測量用の採用閾値を流用せず、診断値として人手レビューに渡す。
clip別 `pose_reprojection_diagnostics` は同じ最終正重みmaskをraw/refinedに適用し、カメラ・選手別に
confidence ≥ 0.3、カメラ前方かつ有限の全関節残差を、観測と教師投影の両方が画像内の群と
いずれかが画像外の群に完全分割する（画像内は `[0,width) × [0,height)`）。
左右hip両方のconfidence ≥ 0.3を満たす観測中心とroot投影の残差も別に集計する。
画像外の残差は削除せず、画像内外で教師weight・学習mask・既存全clip集計を変更しない。
これらは擬似教師の観測整合性の診断であり、モデルの正しさや測定GTの精度を表さない。

producerのsource manifest SHAとdataset実媒体のSHAを照合し、メタデータ同士が一致しても媒体改変は拒否する。
除外指定したclipがdataset manifestに残っている場合も失敗する。raw archiveはcheckpoint SHA・
reference/ball receipt・観測配列を照合するが、完全なproducer identityが保存されていないため
実行設定すべての同一性までは認証できない。この制限はJSONにも記録する。
