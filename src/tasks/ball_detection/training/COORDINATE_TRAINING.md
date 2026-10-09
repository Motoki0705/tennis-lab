# MDD座標モデルの学習準備

poseあり32条件と、MDDのみのquery-only 4条件を同じ経路で準備する。
モデル構造は[モデル仕様](../models/mdd_pose/README.md)、元のプレイ候補規則は
[区間仕様](../data/PLAY_INTERVALS.md)を参照。ここをデータ固定・FPS混合・評価手順の正本とする。

## 対象の選び方

- **poseあり**: 指定したpose datasetのapproved clipだけ。ball storeとpose snapshotの
  clip ID、source/group/camera、split、寸法・scale、frame/PTS、GTとannotation/media identityを照合する。
- **query-only**: 指定したball storeの全clipを母集団にする。poseの有無・生成skip・clip全体の
  球存在率40%を学習対象の判定に使わない。全clipに同じプレイ区間・窓条件を適用する。
- pose生成campaignの40%条件は変更しない。新たなpose推論やレビューを起動しない。
- 全splitに同じ選択規則を使い、元のsplitを維持する。窓が作れないclipは理由付きで記録する。

## FPSの混合

既定は元FPS、1/2、1/4（`frame_steps=[1,2,4]`）を**同じrun**で混ぜる。
FPSを別モデル数として掛けず、モデルの構成数は36のまま。

- どの条件も入力は32枚。native frameの間隔を1/2/4にし、必要な元frame範囲は32/63/125枚。
- まずnative PTS上でプレイ区間を決める。短い内部欠損の接続は0.4**秒**という実時間のまま。
- 各FPSの窓はプレイ区間内に収め、選んだ32枚で存在証拠率50%以上・実測位置教師8枚以上を判定。
  短い区間を反復paddingしたり、参照frameや大きいPTS gapを跨いだりしない。
- 開始stride16は**間引き後のframe数**。native上の開始間隔は16/32/64。各区間の末尾窓も含める。
- RGBを間引いてからMDDを再計算する。pose・教師・maskも同じframe番号で読む。
  窓の先頭MDDは0。RoPEへ渡すのは実PTS秒で、FPSの表示値だけを変えない。
- 学習はFPS間でほぼ等数の窓を使う（端数の割当はepochで巡回）。各FPS内はshuffleし、
  窓を使い切った場合は新たなshuffle cycleで補う。
- 既定の1epochの窓数は、その学習集合の**native-FPS窓数**。`--windows-per-epoch`で明示変更できる。
  同じepoch数でもposeあり／なしでデータ量が違うので、更新数・一意frame数と併せて記録する。

統計・既存WebUIのnative窓提案と、混合FPSの学習窓は区別する。
実験で採用するFPS別窓の正本は以下の生成manifest。

## CPUで準備する

実装のあるworktreeから実行する。ここでは学習・GPU推論・queue登録は行わない。
出力先にはまだ存在しないrun directoryを指定する。

```bash
BALL_CODE_ROOT="$(pwd)"
BALL_REPO_ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
BALL_PREP_ROOT="$BALL_REPO_ROOT/outputs/ball_detection/precompute/mdd_coordinates_36/<run-id>"
.venv/bin/python -m src.tasks.ball_detection.scripts.prepare_mdd_training \
  --ball-store "$BALL_REPO_ROOT/data/ball_detection/ball-mix-v2" \
  --poses "$BALL_REPO_ROOT/data/ball_detection/ball-mix-v2-player-pose-v1" \
  --model-config "$BALL_CODE_ROOT/src/tasks/ball_detection/configs/model/mdd_pose.yaml" \
  --output "$BALL_PREP_ROOT" --frame-steps 1 2 4 --window-stride 16
```

生成物:

```text
<run-id>/
  experiments.json            # 36構成、設定hash、データ、実行引数テンプレート
  models/*.yaml               # 32 poseあり＋4 query-only（pose_pooling:null）
  mdd_pose_windows.json        # poseありのFPS別窓・共通評価ID
  mdd_only_windows.json        # 全母集団のFPS別窓・共通評価ID、pose file依存なし
  ball_snapshot/               # metadata/indexのコピー、JPEG shardのhardlink
  poses/                      # 採用poseとレビューの独立コピー
  pose_source_manifest.json    # 採用時点の元pose manifest
```

RGBは巨大な複製を避けるためhardlinkとし、出力は同じfilesystemに置く。
別filesystemへ黙って全RGBをコピーしない。hardlinkは元storeのpath置換からは独立するが、
同じinodeへの上書きは共有するため、学習readerは最初の読込でshard hashを検証し、その後も変更を検出する。
metadata/index、pose/review、実装sourceのhashを保存する。準備中の入力・実装変更は拒否する。
`PREPARATION_INCOMPLETE`が残る出力は未完了で、学習planとして使わない。

新形式は`ball_coordinate_windows.v1`。既存の`ball_play_windows.v1`は
poseあり・native FPSだけの形式として明示的に読める。新しい形式に推測で置き換えない。

## 学習と評価

`experiments.json`の各`command_template`は、epoch・学習率・seed・新規出力先を
確定してから使う。GPU学習・GPU評価は[共有training queue](../../../../.agents/skills/training-queue/SKILL.md)
経由で実行する。元repoのqueueを共有し、worktreeごとのqueueを作らない。

学習入口は`src.tasks.ball_detection.scripts.train_mdd_pose`。
poseあり／query-onlyの選択はmodel configとmanifestの明示的な組合せで決まり、
不整合時に空poseや別データへfallbackしない。

- 位置lossは単一observedのuvに対するSmoothL1。未確定位置を補間して教師にしない。
- validationではFPSごとに`(clip ID, 実際のframe番号)`で重複除去する。
  窓の中央に近い予測を採用し、同距離なら開始の早い窓を採用する。
- **full / common**両方について、FPS別・source別のframe数、平均・中央値・P95誤差を出す。
  commonはposeありとGT/split/画像identityが一致したapproved subset。
- checkpoint選択の既定は**commonのvalで、FPS別平均誤差を等重み平均**した値。
  `--selection-scope full`も明示選択できる。指定scope/FPSの教師が0件なら、別scopeへ切り替えず失敗する。
- 共通subsetだけで学習する追加のquery-only runは作らない。共通評価でも学習集合・学習量の違いは残る。
- testのsampleは学習とcheckpoint選択に使用しない。次の入口で選択後に明示評価する。

```bash
.venv/bin/python -m src.tasks.ball_detection.scripts.evaluate_mdd_coordinates \
  --checkpoint <absolute-epoch-checkpoint.pt> \
  --manifest <the-same-frozen-window-manifest.json> \
  --output <absolute-new-evaluation.json> --split test --device cuda
```

評価はcheckpointと同じmanifest hashを要求し、MDD設定・モデル状態をstrictに復元する。
新checkpoint形式は`mdd_coordinates.v4`で、実際のcode hash、native RGB/MDD入力契約、データidentity、
checkpoint選択scopeを記録する。準備の際に学習やtest評価を自動起動することはない。

Conv2d＋query-onlyの具体的な入力・損失・予算・実行条件は
[初回学習レシピ](CONV2D_QUERY_ONLY_RECIPE.md)にまとめる。

## 実行時設定・中断からの再開

`--precision fp32|bf16`は学習・validationに適用する。明示的なtest評価も保存された精度を復元し、
評価CLIの`--precision`で変更した場合は実際の精度を結果へ記録する。
旧v2/v3 checkpointはJPEG decoderを固定できないため明示的に拒否する。旧診断は保存済みcommit/bundleで再現する。
BF16は対応CUDAデバイスを要求し、別の精度へfallbackしない。readerは選択したJPEGまたはRGB uint8を返し、
JPEGは指定decoderでRGB uint8へ復号する。モデル内で正規化・輝度・MDDをFP32で計算する。PTS・lossもfloat32を維持する。
optimizerの重み・状態もfloat32で、FP16/GradScaler・gradient accumulationは使用しない。

train/evaluateの両入口は`--num-workers`、`--pin-memory`、`--prefetch-factor`、
`--cpu-threads`を受け付ける。worker内のOpenCV/PyTorchは各1 thread、workerはepoch間で維持し、
clip hashの検証cacheを再利用する。worker間で初回検証cacheは共有しない。
`prefetch_factor`はworkerごとの先読みbatch数であり、高解像度のhost RAMと共有メモリも消費する。

学習はAdamW（weight decay 0.01）、一定学習率、gradient norm上限1.0。
`train.jsonl`に`--log-every` updateごとのobserved-frame加重train loss、LR、gradient norm、速度を保存し、
`metrics.jsonl`にepochごとのvalidation、train loss、学習・評価時間を保存する。
epoch checkpointと`best.json`は一時ファイルからatomicに公開する。

同じコマンドに`--resume <同じoutput内の最新epoch-NNN.pt>`を付けると、次のepochから再開する。
総epoch数は`--epochs`で延長できる。モデル・データ・LR・BS・seed・実行時設定・実装hashが変わる再開と、
過去epochへの巻き戻しは拒否する。optimizer、CPU/CUDA RNG、sampler epoch、更新数、best選択を復元する。
旧checkpointに再開stateがなければ明示エラーになる。epoch途中からの再開は行わず、
中断したepochを直前の完了checkpointからやり直す。途中のtrainログは履歴として残り、
`resume.jsonl`で再開位置を区別する。最初のepochを保存する前に中断した場合は新規runとしてやり直す。
同じseedはGPUでのbit単位の再現性を保証するものではない。

## JPEG復号と入力の先読み

`--jpeg-decoder opencv|nvjpeg`で復号方式を固定する。opencvはCPUでRGBへ変換し、nvjpegはworkerから
`preadv`で必要なJPEG範囲を1つのbufferへ読み、圧縮JPEGだけを渡してCUDAで復号する。nvJPEG内部のhost処理は残る。OpenCVへの自動fallbackは行わない。
復号後はどちらもRGB uint8のモデル入力となり、MDD以降の構造は共通。GT・split・frame番号は変えない。
OpenCV/libjpeg-turboとnvJPEGの画素値は一致しないため、decoder/API/versionをv4 checkpointの
`image_decode`と学習recipeへ記録する。評価は保存設定を復元し、無指定で実装versionが変わる場合は拒否する。
別decoderでの評価は`--jpeg-decoder`で明示し、元・実行時の両契約を結果に残す。

`--image-prefetch`はnvjpeg専用。1 producerが次batchのJPEG復号を別CUDA streamへ送り、現在のモデル計算と重ねる。
順序を維持し、event・record_stream・CPU byte bufferの保持で寿命を管理する。早期終了時もproducerをjoinし、エラーを伝播する。
CPU reader待ちと、GPU復号も含む入力準備完了待ちは別物。`train.jsonl`の`mean_reader_wait_seconds`は
producer内のreader待ち、`mean_input_ready_wait_seconds`はmain内の待ちで、重なりがあるため加算しない。

`--input-verification lazy|upfront`は同じdual SHA-256検証を、初回使用時またはloader開始前に行う指定。
成功結果だけをfork/spawn worker間で共有し、clipごとに一度検証する。
範囲読込のdescriptorにも同じstat identityを要求し、path差替え・EOF・読込中の変更を拒否する。以後もdev/inode/size/mtime/ctimeの変更を拒否する。
新runでは再検証し、検証省略や永続的なtrust cacheは作らない。upfrontのCPU時間は準備費用として記録し、学習速度から除外した場合も総時間へ含める。

## torch.compile

`--compile-mode off|default|reduce-overhead|max-autotune`でCUDAモデルのcompileを明示する。
`off`がCLI既定。モデルをin-placeでcompileし、型判定とstate_dictのparameter名を維持する。
backendはInductor、`fullgraph=True`、`dynamic=False`で、graph breakを許さない。
`--compile-recompile-limit`は既定8。train/eval、最終batch、pose人数のshape違いによる特殊化を含み、
上限を超えた場合はエラーにする。Tensor値をPython boolへ変換する検証はcompiled forwardの外に置く。
学習はautocast下でforward/loss、その外でbackwardするため、遅延compile・backward時も
`backward_pass_autocast="off"`を明示する。エラーを隠してeagerへfallbackしない。

評価は保存されたcompile設定を復元する。`--compile-mode off`で明示的なeager評価もできる。
設定・各epoch checkpoint・評価結果に、compile modeとprocess内のgraph数/graph breakを記録する。
初回compileを含む時間と、warmup後の定常速度は分けて計測する。
GPU実行時は`TORCHINDUCTOR_COMPILE_THREADS=2`でhost RAMを制限し、
`TORCH_LOGS=graph_breaks,recompiles`で意図しない再compileを追跡できる。

## 実装

- `data/temporal_sampling.py`: native play span内のFPS別32枚と境界の契約。
- `data/coordinate_manifest.py` / `coordinate_snapshot.py`: GT共通性の照合とsnapshot固定。
- `data/coordinate_dataset.py`: sampled JPEG/RGB、同じindexのpose/教師/PTS、共有された検証結果を読む。
- `models/mdd_pose/variants.py`: 重複のない36モデルの列挙。
- `coordinate_preparation.py`: 準備transactionと構成・入力・コードの記録。
- `coordinate_sampling.py`: epoch予算を保った均等FPS混合。
- `coordinate_evaluation.py`: FPSごとの一意frame評価、全体／共通／source集計。
- `preprocessing/mdd.py`: 固定FP32 RGB→MDD。モデルのforwardに含める。
- `coordinate_images.py`: JPEG decoder契約とCUDA streamでの先読み。JPEG復号はcompileの外。
- `coordinate_compilation.py`: fullgraph/AMP backward方針と実行記録。
- `coordinate_runtime.py`: 精度、reader並列度、共通のoptimizer update。
- `coordinate_checkpoint.py`: epoch単位の再開とatomicなcheckpoint/best公開。
