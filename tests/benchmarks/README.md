# 実データの診断

通常の単体テストには含めない、実データ・固定bundleでの数値診断です。

## 人物追跡方式のCPU比較

`person_tracking_matrix.py` の `--phase track` → `evaluate` → `report` は
同じ `--repo <main root> --features <run-7 feature root> --report <新規出力先>` を使う。
単一CPU process・torch4/OpenCV1 thread、入力/出力hashと実行identityを固定しcamera単位で再開できる。
比較範囲・主副指標・既知の偏り・パラメータ・推薦規則の正本は
[事前commitしたrun-8プロトコル](../../knowledge/runs/run-i964-tracker-matrix-r8-20260930/protocol.md)。
baselineのLab連結が停止した場合も停止として保存し、候補追跡で補完しない。
`person_tracking_matrix_video.py --report <同出力先>`は固定した最大差5秒窓を3camera動画にする。

`person_tracking_linking.py --phase base` → `kpr` → `report` は
[run 9 addendum](../../knowledge/runs/run-i964-tracker-linking-r9-20260930/protocol-addendum.md)の追加比較。
`--repo`、`--features`、`--kpr`（回収済native特徴root）、`--aflink`（公開重み）、
`--previous`（run 8 matrix）、`--report`（新規出力先）を明示する。
既存6条件のraw主指標一致を確認し、downstream group指標を追加する。
`person_strongsort_parity.py --upstream <別途取得した固定版> --weight <AFLink重み> --report <JSON>`は
ラベルを使わず合成trackletでAFLinkの前処理・学習済み推論の数値互換を確認する。
`person_tracking_linking_report.py --report <run 9 root>`は全表・対応表と推薦規則の結果を出す。
`person_tracking_cam1_diagnosis.py --matrix <run 8 root> --report <診断出力先>`は固定3窓を選び、
camera内の全割当/元検出と診断専用のpose/appearance除去を保存する。
`person_tracking_cam1_video.py --diagnosis <diagnosis.json> --matrix <run 8 root> --output <mp4>`は
全景・遠側拡大・各対応コストを表示し、動画全frameを読み戻す。

`person_kpr_parity.py --upstream <公式repoの固定checkout> --repo <main root> --features <run-7 root> --report <JSON>`
は同じ実dev cropで上流とportのstate key・prompt・native outputをCPU照合する。
`person_kpr_features.py --phase plan --repo <main root> --features <run-7 root> --parity <成功JSON> --report <新規出力先>`
で入力を固定し、同じrepo/reportの`--phase extract`を共有queueから1件だけ実行する。
KPRは各検出の6×512特徴・可視性と元row/box/score/poseを保存し、同frameの他検出poseをnegative promptにする。
検出/pose再推論やtracking/評価は行わず、完了manifestと全値の保存読戻しを記録する。

## Meiji ball holdout

`ball_detection_holdout.py` はball frame storeのMeiji test全体で、ft-e13と
混合FTのvalidation選択checkpointを同じ条件で比較する。checkpointの選択は学習側で済ませる。
既定のtestはvideo_001、63 camera-clip / 36,006 frameで、数が違う入力は停止する。
モデル・入力サイズ・保存された正規化の一致、train/valへのvideo漏れ、全frameの一意性を検証する。

- 元storeのJPEGをcheckpoint入力サイズへINTER_LINEARでresizeし、公開predictorが一度だけ正規化する。
  stride 4 / tail backfill / max-score集約（同点は後窓）、subpixel有効。
  短いclipの末尾反復も出力は元frameだけ。候補もargmaxと同じ窓から採る。
  座標はnormalized×(stored W−1,H−1)÷store scaleで元動画画素へ戻す。
- 主指標はobserved注釈に対するscore >= 0.5 / 20 source pxのrecall。
  欠損数、大誤検出数、受理した全予測のp95、低scoreも含むargmax p95を併記する。
  未解決/未レビューは負例にしない。point_kind別の推定位置とvisibility行は参考値で、
  visibility行は位置のある推定ラベルも含む。top-K recallは閾値なしの候補上限を示す。
- `--poses` は#933の保存済みstore root。media hash・camera・frame数・fps・解像度と
  descriptor/配列のchecksumを検証してCOCO17手首だけ読む。conf >= 0.5の手首と
  注釈球の最短距離 <= 100 source pxをnear_wrist、超過をflightのproxyとする。
  pose/有効手首/球位置の欠落はunknown。poseが存在する部分集合への選択バイアスと
  coverageは`protocol.json`に残す。閾値はCLIで明示変更できる。
- trajectory gateは使わない。storeの720p JPEG経由でもあるため、#932のraw動画＋gateと
  絶対値を直接同一視しない。poseは層別だけに使い、検出器の入力はRGBのまま。

```bash
PYTHONPATH=. .venv/bin/python tests/benchmarks/ball_detection_holdout.py \
    --store <元repo>/data/ball_detection/ball-mix-v1 \
    --poses <元repo>/outputs/player_association/evaluate/meiji_clips/i933-observe-v1-20260927/stores \
    --baseline <ft-e13 checkpoint> --treatment <validation選択checkpoint> \
    --report <新規出力先> --phase preflight --device cpu
```

同じ入力引数で`--phase infer --device cuda`を共有training queueから実行する
（preflightと推論は別の新規出力先を指定）。raw argmax・score・候補・窓の出自と
参照座標/手首距離を圧縮NPZへ保存し、`metrics.json` / `metrics.csv` / `comparison.md`を作る。
全frameの完了前には最終比較を出さない。GPUや元データがなくても
`--phase summarize --report <推論出力先>`だけでchecksum/順序を検証して再集計できる。
条件変更の再集計は出力directoryを複製して`protocol.json`の`metrics`を明示変更する。
元のrunは保持し、変更した条件は別の比較として記録する。

## Pipeline診断

- `ball_detection_evidence.py`: [ball検出証拠](../../src/tennis_scene/pipeline/README.md#ball検出証拠)の
  実clip検証。既定pipelineのball nodeだけを全cameraで実行し、native heatmap・候補・patchを
  `--report/store` に保存する。checksum/型/shapeを検証してdiskからload-onlyで再開し、
  `qualification.json` に各cameraのartifact参照・shape・候補数・gateで非観測になったframeの
  生候補数を記録する。精度比較ではない。GPU実行は共有training queue経由:

  ```bash
  PYTHONPATH=. .venv/bin/python tests/benchmarks/ball_detection_evidence.py \
      --repo <元repo> --clip <構造化clip> --report <検証出力先>
  ```

- `coco17_placement.py`: 身体配置の数値診断。入力と使い方は[motion_alignment](../../src/tennis_scene/motion_alignment/README.md#保存済みデータでの確認)を参照。
- `component_pipeline.py`: 既定`pipeline.yaml`で構造化clipを1本処理する実clip qualification。変更する設定はroot path・device・`execution.ball_detection=load`だけ。
  ball・人物対応は[確認済みデータのimport](../../src/tennis_scene/pipeline/imports/README.md)で埋め、`evaluation.json`の`imported_nodes`に列挙する。
  storeは`--report`配下に作り、clipの`annotations/`へは書かない。import以外の全component実行（sideはimportしたballから`court_side`が決める）、scene export、全段load-only再開を検査する。
  DINO拡張はrepo rootから`build_dino_extension.sh`を実行してrun directory内にbuildし、`PYTHONPATH`に加える。GPU実行は共有training queue経由:

  ```bash
  R=/home/kamimura/projects/tennis-lab; OUT=$R/outputs/tennis_scene/evaluate/<run>
  bash tests/benchmarks/build_dino_extension.sh $R $OUT/dino_extension && \
  PYTHONPATH=.:$OUT/dino_extension/lib .venv/bin/python tests/benchmarks/component_pipeline.py \
      --repo $R --clip $R/data/tennis_multivew/processed/meiji_3cam/dataset/videos/video_000/clips/clip_000 --report $OUT
  ```

  `--dataset <dataset>`を付けると、clipのstoreを`<clip>/annotations/tennis_scene`に作り、
  `generate_pseudo_annotations`で`annotation.json`を公開した後、SLCSとPLCS residualのreaderで読み戻す
  （v1 layoutのdatasetを新しい出力先へ再生成する経路）。`--seed-from <source clip>`はその前に
  clipを作る（mediaはhard link、import入力はcopy、`dataset.json`へ登録。既存clipには書かない）。
  SLCS学習に使うDINO特徴は、その後`python -m src.tasks.slcs.scripts.precompute_dino_tokens data.dataset_root=<dataset>`で作る
  （`paths.output_root`の末尾を`slcs`などtask名にすると、task出力の先頭と衝突してpath contractが停止する）。
- `court_side_clips.py`: 構造化datasetの全clipで、[court side](../../src/tasks/court_side/README.md)を検出器のballと外注注釈のball（`observed`点だけ、参照）の両方から決め、
  判定・停止理由・全仮説のscore・一致と、camera別の検出器ballと注釈ballの一致（`court_side.reprojection_px`以内のframe数）を`--report`の`<--name>.json`（既定`decisions.json`）へ書く。`observe`（GPU、共有training queue経由）は
  court検出・校正とball検出だけを`--report/stores/<clip>`へ実行し、人物・身体は無効にする。`decide`（CPU）は保存済みartifactを読むだけで、
  `--override court_side.<field>=<value>`で閾値を変えて再判定できる。

  ```bash
  R=/home/kamimura/projects/tennis-lab; OUT=$R/outputs/court_side/evaluate/meiji_clips/<run-id>
  PYTHONPATH=. .venv/bin/python tests/benchmarks/court_side_clips.py --repo $R \
      --dataset $R/data/tennis_multivew/processed/meiji_3cam/dataset --report $OUT
  ```
- `player_detection_clips.py`: 標準pipelineのcourt ROI付き人物検出を、既定の選手重みと明示したCOCO重みで比較する（GPU、共有training queue経由）。
  `--repo`・`--dataset`・`--report`を必須とし、両variantのcomponent storeと`comparison.json`を出力する。
  ラベルに無い予測をFPとせず、旧boxとの一致率・既知非選手への反応・未照合件数を分ける。指標の正本は
  [`partial_labels.py`](../../src/tasks/player_detection/evaluation/partial_labels.py)。chat-player-v1 valのAP比較は既存の
  [`player_detection.scripts.evaluate`](../../src/tasks/player_detection/README.md)を`evaluate.split=val`で実行する。
  一括実行用のqueue入口は`player_detection_comparison.sh <元repo> <新しいreport directory>`。
  DINO拡張をrun内でbuildし、検出器の設定をpipeline.yamlから読んで両評価へ渡す。

- `player_association_clips.py`: camera間の人物対応を、ラベル付きclipで評価するための観測。`observe`（GPU、共有training queue経由）は
  court検出・校正と、人物検出・tracking・poseをcameraごとに`--report/stores/<clip>`へ実行する（ball・身体・再構成は無効）。
  trackingが停止したcameraは停止理由と証跡を、完走したcameraは全trackの観測frame数を`observe.json`に残す。
  `--phase sheets`（CPU）は保存済みtrackから、camera別に全trackの等間隔crop（frame番号付き）を`--report/sheets/<clip>/<camera>.jpg`へ描く（ラベル作成の確認用）。
  `--phase labels`（CPU）はreview YAMLの人物割り当てを、trackerに依存しないboxラベルへ変換する（`--review`、保存先はdataset内）。
  ラベルの形式・作成手順・Meiji 3cam のラベルは[player_association](../../src/tasks/player_association/README.md#評価ラベル)を参照。

  ```bash
  R=/home/kamimura/projects/tennis-lab; OUT=$R/outputs/player_association/evaluate/meiji_clips/<run-id>
  bash tests/benchmarks/build_dino_extension.sh $R $OUT/dino_extension && \
  PYTHONPATH=.:$OUT/dino_extension/lib .venv/bin/python tests/benchmarks/player_association_clips.py --repo $R \
      --dataset $R/data/tennis_multivew/processed/meiji_3cam/dataset --report $OUT --clip video_000/clip_000
  ```

  `--phase calibrate`（CPU）は、ラベルの無い観測済みclipの擬似ラベル（camera間のtrackの組を足元距離で分ける）から、
  幾何の`sigma_m`と外観の`slope`・`center`を当てはめて`calibration.json`へ書く（ラベル付きclipを指定すると停止する）。
  `--phase evaluate`（CPU）は、ラベル付きclipを`--config`（既定`src/tasks/player_association/configs/association.yaml`、
  `--geometry-only`で外観なし）で対応付けて採点し、`evaluate.json`とclipごとのコート平面の図（`figures/<clip>.png`）を書く。
  どちらもsideを`court_side_clips.py`の注釈ballによる判定（`--sides`）から読み、trackの外観を`--report/appearance`にcacheする。

  ```bash
  R=/home/kamimura/projects/tennis-lab; OBS=$R/outputs/player_association/evaluate/meiji_clips/i933-observe-v1-20260927
  SIDES=$R/outputs/court_side/evaluate/meiji_clips/i932-detector-v1-20260927/decisions_v2.json
  PYTHONPATH=. .venv/bin/python tests/benchmarks/player_association_clips.py --repo $R --phase evaluate \
      --dataset $R/data/tennis_multivew/processed/meiji_3cam/dataset --observe $OBS --sides $SIDES \
      --device cpu \
      --report $R/outputs/player_association/evaluate/meiji_association/<run-id>
  ```

- `player_detection_disagreements.py`（CPU）は保存済み`--comparison`とdataset内ラベルから、
  未一致の旧boxをIoU・box高・camera・近遠（画像内のbox下端順位）別に集計し、
  `--report`へ旧box/新検出の短い比較動画を書く。旧COCO box由来の偏りがあるため検出recallとは呼ばない。

- `player_association_reserve.py --repo <元repo> --report <新規出力先>`（CPU）は、人物処理の設計・評価・
  擬似ラベル校正の履歴とclip metadataだけを読み、各動画から600frame以上の最長の未使用clipを予約する。
  datasetの`annotations/player_association/unseen_protocol.json`を更新し、旧予約・選定/除外理由・hashをreportへ保存する。
  映像をdecodeせず、ラベルを作らない。未ラベル・調整未完了・評価試行0の予約だけを変更できる。

- `player_detection_far_diagnosis.py --phase preflight --repo <元repo> --comparison <run1/meiji/comparison.json> --report <新規出力先>`
  はrun 3の履歴再現用。2026-09-29のユーザー判断でGPU jobは中止し、高解像度・tileの追加実行は行わない。
  新しいCPU比較は下記`person_selection_cpu.py`を使う。
  は、既存4開発clip・動画/ラベル/重みhash・固定court ROIと未見予約の非重複をCPUで確認する。
  その後`player_detection_far_diagnosis.sh <元repo> <comparison.json> <report>`を1つのGPU queue jobで実行する。
  #937の既定resizeでscore 0.01まで保存し閾値曲線を作る。native 1080、1080/1440/1800/2160/2880/4320の
  probeで全3cameraを通った最大試行サイズ、画像上半分の重複tile＋既定、旧COCOのROI内小box（高さ64px以下）unionを比較する。
  最大値はこのGPU/float32での試行結果で、モデル固有の上限とは称さない。OOMはprobeの証拠として記録し、通常推論の失敗は停止する。
  `diagnosis.{json,csv,md}`にcamera×近遠×variantの旧box一致率・追加既知非選手単位・hit率・ms/frame、
  `missed_old_scores.jsonl.gz`に全未一致選手単位の最高score（IoU≥0.3、ROI前後）、
  `old_vs_best_variants.mp4`に旧boxと開発一致率上位2条件の24秒比較を出す。
  `--phase summarize`は全raw archiveのhashを検証しCPUだけで再集計する（出力済みrunは別directoryへ複製してから使う）。
  ms/frameはscore 0.01の共通forward＋ROI/unionで、動画decode/load/warmup/保存は除く。未ラベル予測数も別記する。
  pipeline設定・重み・既定thresholdの変更、方式の最終比較、未見clipの評価は行わない。

- `person_coco_fullframe.py --phase preflight --repo <元repo> --comparison <run1/meiji/comparison.json> --report <新規出力先>`
  は4 dev clip × 3cameraだけをhash検証する。`timeout 5400 bash tests/benchmarks/person_coco_fullframe.sh <元repo> <comparison.json> <report>`
  を共有GPU queueへ1件登録すると、旧COCO DINOの800/1333、score 0.01の全画面box・scoreをROI前に保存する。
  torch allocatorは6 GiBに制限。camera単位のarchive/hash・進捗・peak allocated/reservedを残し、途中runを上書きしない。
  ROI後の旧storeと区別し、最終のソース比較とCPU閾値sweepは別runで行う。

- `person_selection_refinement.py --previous <run4-report> --report <新規出力先>` はCPUのみで
  FT 0.01 / 保存済みunion / 旧経路のtrackとCLIPを再利用する。`court_linking.py`の領域と断片連結を適用し、
  旧基準・領域だけ・領域＋連結の3段の人物unit/identity、camera×近遠、clip別表を保存する。
  `diagnosis.json`は旧選別で残った隣コートunitのtrack構成・座標・旧/新領域内外を記録する。
  COCO/unionのROI保存差はまだ残るので、全画面COCO完了後の公平な最終比較には代えない。
  camera間対応は再実行せず、person_identities v3の安全策とCLIP既定は変更しない。
  `person_selection_failure_video.py --report <同report>` は2つの隣コート失敗例をFT/union各4秒、
  3camera同期映像・cam0拡大・コート足元図で16秒にまとめる。ラベルは事後の失敗例指定に限る。

- `person_selection_cpu.py --repo <元repo> --report <新規出力先> --phase sources --progress <run3/progress.json>`
  は中止済みrun 3のft_base 12件・ft_1080 11件をhash検証し、保存済みCOCOと比較する。閾値
  0.01/0.02/0.05/0.1/0.3のcamera×近遠表、共通11件の表、ROI内外人数を`sources.{json,csv}`へ書く。
  参照はCOCO由来の旧boxでCOCOに有利。COCOはROI後しか保存されておらず、ROI外の人数は不明。
  FT/COCOの推論時間は計測範囲が違うためJSONの`runtime_scope`を必ず読む。
  同じreportで`--phase tracks`はFT base 0.01・COCO・unionをCPU Ultralytics BoT-SORTへ渡す。
  元scoreは保持し、追跡段の追加score gate/fusionと人物全体の上限は無効。検出row対応を検証し、raw/Kalman両boxを保存する。
  `--phase select`は足元の既存校正・領域・presence fractionで選手候補を選び、上限6を適用する。
  既定CLIP-ReIDもCPUで実行し、#933の区間分割・短い曖昧区間除外・handoffを含むcamera間対応を第2確認に使う。
  未決定は理由を残し、成功結果へ戻さない。旧検出＋旧追跡の保存済み出力もbaselineとして同じ選別に通す。
  `--phase video`はunionのdev clip_000から、全人物を灰色、選手を予測identity色で示す12秒3camera動画を作る。
  `--phase report`はidentity/camera別の表と日本語`report.md`を作り、対応後の足元のcamera間距離（z=0）を第2確認として記録する。
  pipeline既定やencoder比較は変更しない。全人物のpose/外観が未保存のBoT-SORT-style derivativeはこの診断では未評価。

- `person_selection_fullframe.py --phase sources --ft-progress <run3/progress.json> --coco-inference <COCO/inference.json> --report <新規出力先>`
  はFT .01/.02/.05、全画面COCO .05/.10/.30、両者 .30のunionを同じ800/1333・ROI前の条件で比較するCPU診断。
  全24 raw archiveのhash・全12 camera-clip・入力/校正/未見予約の一致を検証する。
  `--phase tracks --max-cameras 1`は同じscore gateなしBoT-SORTを再生し、camera境界の完了hashから再開する。
  旧ROI後COCOを補完や代用に使わない。
  `--phase select --repo <元repo> --max-clips 1` はcameraごとにCLIP/選別を保存し、1 source×clipずつ進める。
  `--phase report` は全7×4結果のhash・全選択断片の観測保持・上限を検証して、camera×近遠・identity・wideの表を書く。
  領域/CLIP/fragment/handoffの定義は[`court_linking.py`](../../src/tasks/person_tracking/court_linking.py)を正本とする。
  `person_selection_fullframe_video.py --report <同report> --source <選んだsource>` は各clipの最大誤り窓とwide/隣コート窓を
  3camera同期で描き、全frame読戻し・hash・窓の選定基準を保存する。ラベルは事後の可視化にのみ使う。

- `person_tracking_dev_features.py --phase plan --repo <元repo> --sources <run6/sources.json> --report <新規出力先>`
  はCOCO全画面0.30の4 dev clip×3cameraだけをhash検証し、重み/入力/未見予約を固定する（CPU）。
  `--phase extract --repo <元repo> --report <同出力先>` は共有queueの1 jobでViTPose＋CLIP→SOLIDERを抽出する。
  全人物rowを保持し、SOLIDERには保存済みposeを使う。モデル選択・追跡比較・GT照合を実行しない。
  allocator上限7 GiB、pose batch4、appearance batch8、外側timeout5400秒を必須とする。
  KPRは別のnative-part特徴入口を使うため対象外。成功は`features.json`、進捗/失敗は`features.progress.json`、NPZはencoder/clip/camera別。
  既存の成功/失敗出力は上書きしない。
