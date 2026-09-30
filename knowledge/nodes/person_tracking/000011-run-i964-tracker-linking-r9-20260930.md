---
id: run-i964-tracker-linking-r9-20260930
type: run
task: person_tracking
sequence: 11
recorded_at: '2026-09-30'
title: offline linking・StrongSORT++・native KPRの固定比較とcam1遠側診断
issue: 964
provider: codex
status: done
config:
  primary: run-8 raw-ID partial-reference IDF1; unchanged
  secondary: downstream court-linked group IDF1; added after run 8 results
  detector: COCO full-frame .30
  dev_clips: 4
  cameras: 3
  conditions: 9
  cpu_threads: 4
  gpu_jobs_this_run: 0
  pipeline_default_changed: false
metrics:
  recommended_raw_idf1: 0.9301048427358961
  recommended_group_idf1: 0.936645032451323
  recommended_pair_f1: 0.9485209574579283
  kpr_tracker_raw_idf1: 0.8692494553104104
  kpr_tracker_group_idf1: 0.9399213643535098
artifacts:
  run_dir: knowledge/runs/run-i964-tracker-linking-r9-20260930
  output_dir: /home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/tracker_matrix/i964-linking-r9-20260930-v3
parents:
- run-i964-tracker-matrix-r8-20260930
- run-i964-kpr-native-features-r8-20260930
relations: []
papers: []
tags: []
date: '2026-09-30'
repro:
  commit: ebaaa0f1
  protocol_commit: ac1957ad
  command: PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
    .venv/bin/python tests/benchmarks/person_tracking_linking.py --repo /home/kamimura/projects/tennis-lab
    --features /home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/dev_features/i964-features-r7-20260930
    --kpr /home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/dev_features/i964-kpr-r8-20260930
    --aflink /home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/tracker_matrix/i964-linking-r9-20260930/resources/AFLink_epoch20.pth
    --previous /home/kamimura/projects/tennis-lab/outputs/person_tracking/evaluate/tracker_matrix/i964-matrix-r8-20260930
    --report <new-output> --phase <base|kpr>
---

## 推薦と下流指標の不一致

**事前に固定したraw-ID主指標による候補内推薦はStrongSORT++ / CLIP-ReID。pipeline既定は変更しない。**
[全9条件・camera×near/far・camera間3encoderの表](../../runs/run-i964-tracker-linking-r9-20260930/report.md)と
[人物→予測ID対応](../../runs/run-i964-tracker-linking-r9-20260930/correspondence.csv)を保存した。
主指標は0.930105、12/12camera完走、switch10/fragment12。Deep OC-SORT+pose/CLIPの0.895717、14/31から改善した。
cam1 farも0.703617→0.893259と改善する。ただし新検出+旧追跡の0.937610には届かない。
StrongSORTには独自pose costを追加していない。論文方式の比較であり、pose併用を確認した方式という意味ではない。

推薦をそのまま既定採用の合格とはしない。追加副指標group IDF1は推薦0.936645に対して
new+old 0.943699、camera間pair F1も0.948521に対して0.979717（Deep/CLIPは0.967598）。
cam2 farのgroup IDF1は0.8377でnew+old 0.9993より低い。主指標と下流指標が支持する順位は一致していない。
new+oldは11/12完走でclip_007/cam0の1,202 player unitをFNに数える。同じ停止規則の下で
推薦は保持19,499 vs18,997だが、単純な同一coverageでの改善とはしない。全9条件の分母20,558、既知非選手4,454を維持。
旧box/旧track由来ラベルのbaselineへの偏り、設計済み4 dev、singleのみという制約は残る。

## offline linkingを揃えて見えたこと

[addendum](../../runs/run-i964-tracker-linking-r9-20260930/protocol-addendum.md)は**新結果より前にac1957adでcommit/push**し、
[Issueにも投稿](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5903288749)した。
raw-ID主指標・IoU .5/1対1/人物unit/タイ規則はrun 8のまま。group IDF1はrun 8結果後に追加した副指標と明示する。
旧6条件のraw指標は全camera/全96層で一致し、旧結果を改変していない。

Deep/CLIPに**同じ旧Lab再連結**を加えるとraw0.743561/group0.748106、9/12完走となった。
clip_007/cam0、clip_001/cam0・cam2が複数候補で停止したためで、曖昧なリンクを選んで成功にしていない。
成功cameraでは断片を結ぶ例があるが、これをデータ全体に安全に適用する根拠は得られなかった。
StrongSORT++/CLIPのAFLinkは2 tracklet結合、GSIは全人物1,312 box（選択track上247）を補間した。
KPR版は28結合、824 box（同114）。補間は別mask/row=-1を持ち、主指標や選別の実観測へ昇格させていない。
そのためGSIによる主指標FN/fragmentの改善とは解釈しない。動画では`GSI ... synthetic`と別表示する。

## ライセンス・実装検証・KPR

[StrongSORT notice](../../../src/tasks/person_tracking/strongsort_NOTICE.md)と
[暫定判断](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5903293645)のとおり、
GPL公式コードをvendorせず論文から推論再実装した。AFLink重みの独立した条件は未確認なので、
MITとみなさず再配布せず、今回のローカル研究比較に限定する。再実装が重みの条件も解決したとは扱わない。
AFLink SHA-256は`b35cbeddd3acc48fece820bd640640e6bfb1f5fbf570aa79af26c6a38958daa4`。
[144 key/合成5組の前処理・推論parity](../../runs/run-i964-tracker-linking-r9-20260930/aflink-parity.json)は最大差0。
StrongSORT全体の公式parityではない。NSAは事前指定した論文式9（covariance×(1-score)）を使い、
公式コードのstdに係数を掛けてsquareする式との差をnoticeで明示した。学習はしていない。

初期実装の潜在height正値検査が未観測trackを過剰に停止したため、
[失敗を保存](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5903522204)し、
[公式への合成入力検査に基づき修正](https://github.com/Motoki0705/tennis-lab/issues/964#issuecomment-5903568735)した。
負の潜在状態を保持し通常max_age30で期限切れにする。実検出boxとは扱わず、非有限は停止する。
v1/v2は調査用として保持し、正式表はv3。閾値・ラベル・選別を探索して直していない。

KPR方式はCLIP条件のraw-ID順位を凍結した`kpr_method.json`でStrongSORT++に決めてから実行した。
KPR trackerはraw0.869249/group0.939921。後者ではCLIPよりよいが、主指標の順位を事後変更しない。
HL3 noticeと重みhashを維持し、共通可視partの**Euclidean距離の平均**、part別EMA、区間の可視part平均を使う。
単一cosineへのflattenはしない。camera間KPRは固定CLIP尺度への`1-distance`転用で大半がambiguous_playersとなり、
推薦tracker上では0/4clip決定（pair F1未定義）。例としてclip_000の同人物を支持する幾何4.192に対し
part distance0.480→similarity0.520、外観scoreは下限-4へ飽和する。尺度転用の不適合を含むため、
KPR一般の劣位とはいえない。別の未ラベルデータによる距離校正が必要だが今回は行わない。
SOLIDERは今回もcamera間の最終対応がCLIPと同一だった。3encoderの最適校正同士の比較ではない。

## cam1 farの原因診断

[全trace・反実仮想](../../runs/run-i964-tracker-linking-r9-20260930/diagnosis.json)、
[欠測内訳と競合例](../../runs/run-i964-tracker-linking-r9-20260930/diagnosis-causes.json)を保存。
全4cam1の再生はrun 8のID・box・元row・maskに一致した。farの欠測419 unitは**全件にIoU≥.5の元検出がある**。
70 unitはtrackerがemitせず、349 unitはemit後に固定選別で落ちていた（そのうち289がclip_001）。
検出器の取りこぼしではなく、追跡と選別までの情報の混在が主因だった。

- clip_000: frame476、row975/976は同じ遠側選手をIoU0.822で重複検出し、ID2/4が競合する。
  frame488と523でID4へ移り、途中でID2へ戻る。frame523では外観はID2へのcosine0.960がID4への0.949より高いが、
  IoUは0.671 vs0.909、総scoreは1.471 vs1.734となりID4を選ぶ。min_hits3が復帰直後の欠測も生む。
  470 unitの「ID誤り」は同じ人物のID断片化をglobal ID割当で数えたもの。470回の人物取り違えではない。
- clip_001: frame963、重なる別人の検出row3188がID2に強く一致（IoU0.863、総score1.703）。
  選手のboxは元々非選手を追っていたID4へ移る。選手boxのCLIPはID2へ0.976、ID4へ0.731と正しい旧IDを支持するが、
  IoUと全体Hungarianの割当で反転した。ID4の長い非選手区間によりコート滞在選別から落ち、選手の289 unitを失う。
- clip_013: frame176、row354/355が同じ遠側選手をIoU0.528で重複検出しID2/4を作る。
  frame184/186/187で競合。45–65 px程度の小さなboxの形状変化がIoU/pose costへ現れる。

小cropが原因という単独説明は支持しない。多くのfar boxは約50–62pxでも可視joint数中央値16–17、
隣接frameのCLIPは高い。診断専用のposeゼロ/appearanceゼロでも全clipを一律に回復せず、
clip_000 far raw未選別IDF1は0.5195→0.5013/0.5184、clip_007は0.7290→0.4630/0.7382だった。
poseはclip_007ではむしろ追跡を支える。これらは全raw観測の原因分離で、候補ランキングへ混ぜていない。
重複boxの扱い・motionと外観の競合が次の調査候補だが、今回の検出/選別/defaultは変更していない。

## 動画と検証範囲

[3cameraレビュー動画](https://github.com/Motoki0705/tennis-lab/releases/download/campaign930-i964-r9-review/review.mp4)
はnew+old対StrongSORT++/CLIP、各clipの最大差5秒窓、計20秒/300frame。
raw ID・group ID・両IDF1・switch/fragment・baseline停止・GSI syntheticを表示する。
[cam1診断動画](https://github.com/Motoki0705/tennis-lab/releases/download/campaign930-i964-r9-review/cam1-diagnosis.mp4)
は3窓/15秒/225frame、全景・拡大・元row/ID・IoU/外観/poseの寄与を表示する。
窓は誤り数上位の非重複窓（clip_000 521–820、clip_001 963–1262、clip_013 0–299）で固定した。
[verification](../../runs/run-i964-tracker-linking-r9-20260930/verification.json)に全108 camera-conditionと
元検出row・group出力・GSI mask・動画読戻し・hashの照合を記録した。診断traceを含むbundleと
失敗ログは保存し、GPU0件、予約未見0件。参照sideは既存注釈ball由来で全pipeline qualificationではない。

次はorchestratorが主/下流指標と動画を提示して既定trackerを確認し、決定後にpipelineと#935共通人物処理へ接続する。
clip_000全pipelineと、調整凍結後の予約未見1回/3camera評価は未実施であり、#964全体は継続する。
