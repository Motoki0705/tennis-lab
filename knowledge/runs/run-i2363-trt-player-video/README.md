# 動画一括のプレーヤー3D再構築をTensorRT FP16で計測する

[#2363](https://github.com/Motoki0705/tennis-lab/issues/2363)の評価条件は [condition.json](condition.json) に固定する。meiji_4camの `video_000` 全69クリップで横断バッチを有効にし、人物対応は各side 2人、コートの向きは各clipのplayer証拠、ボール検出と再構成は無効とする。コートには[承認済みの共通Hと線offsetによる診断](../../nodes/court_detection/000078-run-court-video-drift-lines-20261117.md)を用いる。

## バッチサイズの根拠

| モデル | B | 単体throughput | Peak VRAM増分 | 単体精度 |
| --- | ---: | ---: | ---: | --- |
| ViTPose-H | 8 crop | 142.43 crop/s | 1,699 MiB | xy差P95 0.024687 px、最大0.302900 px、mask差0 |
| HMR2 | 16 | 290.65 crop/s | 1,597 MiB | cosine P05 0.999994、相対L2 P95 0.003534 |
| DINO GridSample | 1 | 13.33画像/s | 1,627 MiB | IoU P05 0.929859、score差P95 0.017161。単体基準未達 |
| CLIP-ReID | 16 | 2,358 crop/s | 435.1 MiB | cosine P05 0.999987、最近傍ID 236/236一致 |
| GVHMR | 4 | 1,783.6系列/s (L=145)、1,607.8 (L=177) | 287.1 MiB | カメラ座標COCO-17 P95 8.78 mm (L=145)、8.99 mm (L=177) |

画像系のthroughputは計測した完全バッチの値。ViTPoseのcrop数はflip前で入力Bは16になる。ViTPose、HMR2、CLIP-ReIDは各FP16掃引のthroughput最大の候補。DINOはB=2のthroughput差が小さく、B=4では欠落が増えたのでB=1を使う。GVHMRの単体値はB=1〜4・L=145〜177のprofile。全クリップ用profileは、最大のsampled frame数に合わせてL=2〜1055とし、長さの等しい系列をまとめる。VRAM値は、各先行runでのNVMLのidleからの増分で、全体のPeak VRAMは別に測る。 [画像3モデル #2336](../../nodes/tennis_scene/000080-run-i2336-vision-batch-sweep.md)・[CLIP-ReID #2335](../../nodes/tennis_scene/000079-run-i2335-trt-batch-sweep.md)・[GVHMR #2355](../../nodes/tennis_scene/000082-run-i2355-gvhmr-pred-cam-clamp.md)

## 再現

共有 `.venv` にTensorRTのoptional依存がある環境を使う。DINOとDINOv3のsubmoduleを用意する。指定worktreeのrootから、元datasetと先行成果物の実パスを渡して共有training queueへ登録する。

```bash
bundle=knowledge/runs/run-i2363-trt-player-video
queue=.agents/skills/training-queue/scripts/training_queue.sh
printf -v command '%q ' env "PYTHONPATH=$PWD" .venv/bin/python -B "$bundle/run.py" \
    i2363-qualification-v1 --dataset "$MEIJI_DATASET" representative \
    --phase2 "$PHASE2_TENSORRT" --court "$COURT_EVALUATION_ROOT"
bash "$queue" add "$command" --name i2363-trt-qualification-v1 --issue 2363 \
    --resource all --provider codex --session "$CODEX_THREAD_ID"
```

`PHASE2_TENSORRT` は #2362 の `outputs/tensorrt`、`COURT_EVALUATION_ROOT` はrun 77・78の `outputs/tennis_scene/evaluate`。DINOはそのままのplan、ViTPose・HMR2・CLIP-ReIDは保存済みONNXと本番buildコマンドを使う。GVHMRは本番export・build・qualificationコマンドを使う。エンジンの識別と各コマンドの終了値を、worktreeの `outputs/i2363/<tag>/` に保存する。

初回は5モデルのエンジン準備が完了した。GVHMRは24 shapeでONNXの全出力がビット一致し、B=1/4 × L=2/177/1055の6 shapeでTensorRT出力の有限性とFP16層精度を確認した。代表clipはHydraが絶対パスの `progress_directory` を拒否し、推論前に停止した。コマンド記録がGVHMRの計測JSONを上書きしたため、終了値・生の層情報・ログから初回の証拠を保存した。 [初回の終了状態とエンジン識別](qualification-v1.json)

再試行ではHydraへの出力子パスを相対にし、コマンド記録を `*.command.json` に保存する。初回のエンジンをmanifestとSHA-256で再照合し、そのplanでGVHMR単体検証と代表clipを実行する。

```bash
printf -v command '%q ' env "PYTHONPATH=$PWD" .venv/bin/python -B "$bundle/run.py" \
    i2363-qualification-v2 --dataset "$MEIJI_DATASET" representative \
    --prepared outputs/i2363/i2363-qualification-v1 --court "$COURT_EVALUATION_ROOT"
bash "$queue" add "$command" --name i2363-trt-qualification-v2 --issue 2363 \
    --resource all --provider codex --session "$CODEX_THREAD_ID"
```

代表clipは #2362 と同じ `video_000/clip_037`。同一の映像・pass・profileを用い、ホスト計時、同期付き計時の順に実行する。各条件は別プロセスでcomponent cacheが空の状態から開始する。OSのファイルcacheを共有する。資源記録は両条件で0.5秒間隔とし、OMP・MKL・OpenBLASのスレッド上限を2に固定する。全件評価はホスト計時とし、`--progress` で実際に通ったGVHMRの全shapeをORTで照合する。元datasetのmanifestとmediaを評価用ディレクトリへ複製し、評価結果を同ディレクトリへ公開する。

## 代表クリップの結果

`clip_037` は両条件で `generated`。38 nodeを実行し、各選手で3D関節の有効frameは290/292、root・SMPLは289/292だった。38件のcomponent参照と318個のcomponent/scene exportファイルを照合し、artifact ID、execution key、ファイルのSHA-256が一致した。

| 計時用CUDA同期 | CLI wall (秒) | 記録区間 (秒) | CUDA同期span (秒) | Peak VRAM (GiB) | Peak RSS (GiB) | 資源採取中央値 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| なし | 739.79 | 722.96 | 0.176 | 1.76 | 3.67 | 5.79 |
| あり | 899.28 | 878.64 | 141.009 | 1.76 | 3.63 | 8.87 |

同期ありのCLIは159.49秒（21.56%）長かった。同一clipで順に1回ずつ測った値で、OSのファイルcacheや他プロセスの負荷は固定していない。Peak VRAMはNVMLのdevice使用量、Peak RSSはプロセスの最大RSS。WSL全体の空きメモリは計測中に最小22.74 GiBだった。両条件で5モデルのforwardを確認した。この条件で全件評価を行う。

GVHMRは24 shapeでONNX書換え前後の出力がビット一致した。B=1/4 × L=2/177/1055の6 shapeで出力が有限であり、各shapeで110演算、加算mask付きfused MHA 12層、K=2のFP16積和を照合した。代表クリップで実際に通ったshapeはB=4、L=145。 [代表検証の指標と層精度](qualification-v2.json)

ホスト計時の代表clipだけで86,924 spanが記録された。全クリップ実行では、時間窓ごとに完了spanをディスクへ退避し、最終 `recording.json` に全spanと集計を書き出す。 [記録形式](../../../src/utils/inference_recording.md)

## 全クリップ実行

```bash
printf -v command '%q ' env "PYTHONPATH=$PWD" .venv/bin/python -B "$bundle/run.py" \
    i2363-all-v1 --dataset "$MEIJI_DATASET" all \
    --prepared outputs/i2363/i2363-qualification-v2
bash "$queue" add "$command" --name i2363-trt-all-v1 --issue 2363 \
    --resource all --provider codex --session "$CODEX_THREAD_ID"
```

全69件を1 passで試行し、55件が `generated`（scene statusは全件 `ok`）、14件が `failed`、clip単位の `blocked` は0件だった。失敗14 nodeの下流99 nodeは `blocked`。全対象が判定され、記録の保存も完了した。queue終了値1はclip失敗を伝えるCLIの契約による。生成55件はsceneのlineage・保存ハッシュ・annotationの有効数と実maskを照合した。 [全69件の状態・診断](clips.csv)・[4カメラ別の追跡・選手選別](camera-tracks.csv)

5モデルは固定planを各1回loadし、実行時のengine SHA-256が準備時のmanifestと一致した。ViTPose、CLIP-ReID、HMR2のバッチ充填率は99.98%、99.96%、99.86%で、clipをまたいだバッチはそれぞれ141,106、70,563、3,759回。DINOは入力184,912画像をB=1で実行した。GVHMRはB=1〜4、L=2〜1054の67 forwardで使った64 shapeを、seed固定の入力を使ってORTで照合した。ONNXの全出力が書換え前後でビット一致した。内訳はB=1が12、B=2が3、B=3が1、B=4が48 shape。L=177を含むFP16層精度と境界の有限性は準備時に照合している。 [配置したplanとprofile](qualification-v1.json)・[全件のGVHMR shape照合](gvhmr-all-video-shapes.json)

### 時間と資源

再構築CLIのwallは43,858.46秒（12時間10分58秒）、記録区間は43,346.08秒。CLIと記録区間の差には起動・集計・JSON書き出しが含まれる。全件のGVHMR ORT照合は再構築とは別に85.12秒。NVMLでのPeak VRAMは1.76 GiB、プロセスのPeak RSSは6.16 GiB。WSLは総メモリ39.17 GiB、空きの最小25.72 GiBで、16 GB VRAM・40 GB WSLの条件内だった。91回のspan退避を行い、一時ディスク使用のピークは0.66 GiB、完成したtraceは6.57 GB。資源採取83,917回の採取所要時間の中央値は3.63 msだった。 [計測値と全stage・node・モデルの集計](performance.json)

wall配分は各時刻を最深spanへ帰属する。CPU秒とGPU稼働区間率は同じstageの重複区間をまとめて算出する。stage間・モデル間では重なるので、表のCPU秒は個別に読む。GPU稼働区間率はNVMLで使用率0%超だった観測区間の割合で、他プロセスの活動を含む。video pass内ではGPUが観測時間の77.67%で稼働し、使用率の時間平均は54.80%だった。 [記録形式と集計方法](../../../src/utils/inference_recording.md)

| Stage | wall配分（秒） | スレッドCPU秒 | プロセスCPU秒 | GPU稼働区間率 |
| --- | ---: | ---: | ---: | ---: |
| postprocess | 19,669.40 | 19,903.98 | 23,082.32 | 70.27% |
| transfer | 9,549.66 | 9,534.92 | 13,011.48 | 99.74% |
| preprocess | 6,495.86 | 7,076.42 | 24,734.57 | 88.51% |
| inference | 2,221.07 | 2,200.15 | 5,162.27 | 99.33% |
| decode | 1,296.48 | 735.96 | 6,444.41 | 92.33% |
| publication | 1,235.00 | 1,004.29 | 1,040.02 | 9.28% |
| progress_write | 556.95 | 223.10 | 225.67 | 3.53% |
| artifact_load | 428.96 | 364.16 | 366.04 | 2.70% |
| video_read | 35.43 | 29.53 | 37.39 | 69.37% |
| cuda_sync | 0.16 | 0.14 | 0.16 | 9.51% |

decodeはこの条件の画像系横断バッチでは同期readで、`decode_queue_wait` は発生しなかった。GPU出力をホストで読む区間には先行GPU処理の待ちが含まれる。transferのうちViTPoseの `heatmap_to_host` が8,390.75秒（87.86%、166,637回）で、GPUの演算待ち、FP32化と転送の時間がこのspanに入る。

| モデル | wall区間和（秒） | スレッドCPU秒 | プロセスCPU秒 | GPU稼働区間率 |
| --- | ---: | ---: | ---: | ---: |
| DINO GridSample | 19,111.52 | 19,006.00 | 24,891.48 | 99.59% |
| ViTPose-H | 11,530.38 | 11,255.10 | 27,229.42 | 99.26% |
| CLIP-ReID | 1,268.13 | 1,237.17 | 3,528.94 | 95.51% |
| HMR2 | 664.98 | 534.09 | 2,939.99 | 85.70% |
| GVHMR | 121.59 | 120.14 | 122.53 | 90.35% |

モデルのwall区間和にはラベルが付いた前後処理、転送やモデルのload/unloadが入る。CPU側のasync enqueueを囲む `inference` もホスト時間で、GPU演算時間の推定にはCUDA同期ありの代表計測を参照する。

### GPU遊休中のCPU処理

同一スレッド内で子spanを差し引き、GPU遊休区間と重なる分のスレッドCPU秒を推定した。同じ操作の4カメラ分は合算した。三角測量が約72.9分で首位。複数viewのある関節点ごとに候補生成・再投影での検証・最小二乗再最適化を行う既存経路が、GPUを空けたまま主スレッドを長く占有していた。 [node・操作ごとの順位と重複時間](performance.json)・[三角測量の実装](../../../src/utils/geometry/triangulation.py)

| 順位 | 操作 | 遊休中の推定CPU秒 | 改善候補 |
| ---: | --- | ---: | --- |
| 1 | プレーヤーの三角測量 | 4,371.73 | 同じview組の線形初期解と候補の再投影をまとめ、現行のrobust fitが必要な点を測る |
| 2 | カメラ内の人物追跡 | 775.97 | フレームごとの距離・pose更新と割当の内訳を測り、入力をまとめて評価する |
| 3 | 入力の組み立て | 670.15 | 共有court pathの4 cameraで繰り返す切り出しと入力組み立てを確認し、同じ入力で再利用できる部分を測る |
| 4 | scene公開の `ClipStore.record_export` | 530.43 | アーカイブの圧縮・保存と整合性ハッシュの所要を分解して、出力の検証契約を保ったまま書き出しを計測する |
| 5 | 身体の配置 | 371.68 | frame・関節の残差計算と時系列solverの所要を分ける |
| 6 | 中間成果物の `ClipStore.load` | 357.80 | 同じartifactの反復read・ハッシュをnode別に測り、メモリ上限内の再利用を検討する |

全体の待ちの短縮候補にはViTPoseのホスト読出しもある。まずCUDAイベントで先行GPU演算と転送を切り分けると、転送量の削減で得られる時間を見積もれる。

### 出力品質

55件に合計34,772 frame・4選手分のsceneがある。各値は生成済みclipを合算した。失敗clipの3Dは未発行で、全clipを分母とした場合の生成率は55/69（79.71%）。追跡IDと有効maskはパイプライン内の観測・推定に基づく。人物IDの正しさを保証する人手ラベルによる検証は行っていない。 [clip別の有効数・欠測理由・人物対応診断](clips.csv)

| Mask | 有効数 / 全数 | 有効率 | 欠測率 |
| --- | ---: | ---: | ---: |
| 人物観測 | 138,472 / 139,088 | 99.56% | 0.44% |
| 3D関節点 | 2,270,935 / 2,364,496 | 96.04% | 3.96% |
| root・heading（各） | 133,479 / 139,088 | 95.97% | 4.03% |
| SMPL | 133,235 / 139,088 | 95.79% | 4.21% |

全276 camera-clipの追跡には安定IDが3,999件（カメラごとに6〜47、中央値13）あった。選手選別は1,134 group（2〜6、中央値4）を保持した。保持した元trackの観測は659,568/1,319,003件。生成clipの人物対応には13〜21 candidateがあり、clip別の最小decision marginは1.64〜20.67（中央値10.54）。複数trackが同じ人物IDを争いhandoffで除外された観測は10 clipで計57 frameだった。 [追跡ID・観測数・選別結果](camera-tracks.csv)・[margin・各選手の対応camera数](clips.csv)

| 失敗した判定 | clip | 根拠 |
| --- | --- | --- |
| 人物対応の選手境界でrunner-upが近すぎる | 002, 004, 014, 018, 023, 028, 036, 038, 039 | 次点の在コートframe数が採用境界の50%を超えた。028は1,254対1,286、039は831対871 |
| 人物対応で正側の選手不足 | 020, 058 | `y>0` のidentityが各1人。必要人数は2人、最低在コートframeは163/191 |
| 人物対応の長いsegmentで割当が未決 | 049, 060 | margin < 1.0の長いsegment間の判断が6件/4件。最小0.314/0.649 |
| コート向きの幾何支持不足 | 012 | player証拠470 frame。最良hypothesisの支持率が0.19574で必要な0.2に届かなかった |

生成済みの外れ値はSMPL有効率90%未満の6件を選んだ。3D関節点の欠測93,561点のうち92,461点は `INSUFFICIENT_VIEWS`。rootの欠測5,609 player-frameでは4,520が `INSUFFICIENT_JOINTS`、1,067が `INSUFFICIENT_VIEWS` だった。 [失敗の例外・欠測コード・割当view数](clips.csv)

| Clip | SMPL有効率 | 対応・欠測の観測 |
| --- | ---: | --- |
| 005 | 74.28% | P3はcam0でのみ296/314 frameに割当。P3のroot・SMPLは0/314 |
| 022 | 74.57% | P1とP3の割当が1視点のみのframeは374、678。rootは667/1044、363/1044 |
| 001 | 74.60% | P3はcam1でのみ559/561 frameに割当。P3のroot・SMPLは0/561 |
| 032 | 74.67% | P3はcam2でのみ449/451 frameに割当。P3のroot・SMPLは0/451 |
| 064 | 76.54% | P1は1視点のみ939、2視点171 frame。P1のroot・SMPLは168/1252 |
| 019 | 82.43% | P1は1視点のみ388、2視点179 frame。P1のroot・SMPLは178/569 |

判定条件・view不足と欠測の関係は保存した結果から確認した。FP16に起因するかどうかはこの試験から判定できない。

### 保存先と可視化

worktreeの `outputs/i2363/i2363-all-v1/` に評価用dataset、公開scene、実行ログ、資源のraw traceを保存した。E側の実体は `/mnt/wsl/tennis-data/home/kamimura/projects/tennis-lab/.worktree-outputs/parent-2278-issue-2363-primary/i2363/i2363-all-v1/`。 [jobと入力の識別情報](full-video.json)

| 内容 | 上記の出力rootからのパス |
| --- | --- |
| 再構築CLIのログと終了状態 | `all/pipeline.log`・`all/outcome.json` |
| 全件の進捗・計測 | `all/progress/d1df39596d4f47f1896cb535191e7040/progress.json`・`recording.json` |
| 各clipの公開sceneと中間成果物 | `inputs/dataset/videos/video_000/clips/<clip>/annotations/tennis_scene/` |
| 集計CSVとJSON | `analysis/` |
| 代表・外れ値の可視化21本 | `review/<clip>-<status>.mp4` |

`clip_037-generated.mp4` は4 cameraの選手IDと同じframeの3D骨格・有効選手数を表示する。低coverageの6件では同じ表示で欠測を見る。停止した14件の動画は4 cameraの選別trackと取得できた暫定cluster ID、停止理由を表示する。元の60 fps相当の3 frameごとを再生し、MP4のframe数・解像度・SHA-256を記録した。 [動画一覧と検証値](videos.json)

保存済みの結果からCPUだけで集計・可視化を再生成できる。

```bash
bundle=knowledge/runs/run-i2363-trt-player-video
run=outputs/i2363/i2363-all-v1
PYTHONPATH="$PWD" .venv/bin/python -B "$bundle/analyze.py" "$run" "$run/analysis"
PYTHONPATH="$PWD" .venv/bin/python -B "$bundle/visualize.py" "$run" "$run/review" \
    clip_037 clip_001 clip_005 clip_019 clip_022 clip_032 clip_064 \
    clip_002 clip_004 clip_012 clip_014 clip_018 clip_020 clip_023 clip_028 \
    clip_036 clip_038 clip_039 clip_049 clip_058 clip_060
```
