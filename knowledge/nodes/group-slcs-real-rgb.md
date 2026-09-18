---
id: group-slcs-real-rgb
type: group
title: '実RGBのSLCS: Meiji・broadcast教師と入力欠損比較'
members:
- run-slcs-blcs-broadcast-e60-v1
- run-slcs-blcs-meiji-baseline-eval
- run-slcs-blcs-meiji-e60-v1
- run-slcs-blcs-meiji-finetuned-eval
- run-slcs-meiji-v6-tracking-qc
- run-slcs-meiji-observation-mask-probe-v1
- run-slcs-meiji-rgb-features-v2
- run-slcs-plcs-broadcast-e60-v2
- run-slcs-plcs-meiji-foot-e60-resume-v1
- run-slcs-plcs-meiji-foot-e60-resume-v2
- run-slcs-plcs-meiji-foot-e60-resume-v3
- run-slcs-plcs-meiji-foot-e60-selected-test
- run-slcs-plcs-meiji-foot-e60-v1
- run-slcs-plcs-meiji-real-final-eval
- run-slcs-real-court-probe-v1
- run-slcs-real-court-probe-v2
- run-slcs-real-court-probe-v3
- run-slcs-real-geometry-v1
- run-slcs-real-infer-pilot-v1
- run-slcs-rgb-pilot-augmented-e60-v1
- run-slcs-rgb-pilot-augmented-e60-v2
- run-slcs-rgb-pilot-augmented-selected-conditions-v2
- run-slcs-rgb-pilot-baseline-e60-v1
- run-slcs-rgb-pilot-baseline-e60-v2
- run-slcs-rgb-pilot-baseline-selected-conditions-v2
- run-slcs-rgb-pilot-gpu-smoke-v1
- run-slcs-rgb-pilot-gpu-smoke-v2
- run-slcs-rgb-pilot-gpu-smoke-v3
- run-slcs-vitpose-precision-v1
- run-slcs-meiji-temporal-probe-v1
- run-slcs-meiji-checkpoint-hash-probe-v1
- run-slcs-meiji-checkpoint-hash-probe-v2
- run-slcs-meiji-checkpoint-hash-probe-v2b
- run-slcs-meiji-checkpoint-hash-probe-v3
- run-slcs-meiji-checkpoint-hash-cpu-v1
- run-slcs-meiji-checkpoint-hash-gpu-control-v1
- run-slcs-known-memory-cpu-audit-v1
- run-slcs-meiji-root-support-probe-v1
- run-slcs-meiji-inference-repeatability-v1
- run-slcs-meiji-root-support-boundaries-v1
- run-slcs-meiji-v8-root-replay-v1
- run-slcs-meiji-v8-root-replay-v2
- run-slcs-meiji-observation-review-v1
- run-slcs-meiji-observation-review-v2
- run-slcs-ball-gradient-probe-v1
- run-slcs-ball-train-mean-v1
- run-slcs-meiji-observation-review-v3
- run-slcs-meiji-observation-review-v4
- run-slcs-meiji-canonical-association-check-v1
- run-slcs-meiji-observation-review-v5
- run-slcs-meiji-court-visual-qc-v1
- run-slcs-meiji-baseline-line-audit-v1
- run-slcs-meiji-baseline-line-audit-v2
- run-slcs-meiji-v8-observe-v1
- run-slcs-meiji-v8-feature-reuse-v1
- run-slcs-meiji-v8-observe-repair-v1
- run-slcs-meiji-court-crop-comparison-v1
- run-slcs-meiji-v8-features-missing-v1
- run-slcs-meiji-video-alias-audit-v1
- run-slcs-meiji-v9-court-v1
- run-slcs-meiji-v9-observation-reuse-v1
- run-slcs-meiji-v9-feature-reuse-v1
- run-slcs-meiji-read-stream-capture-v1
- run-slcs-host-storage-audit-v1
- run-slcs-meiji-v9-observation-reuse-v2
- run-slcs-meiji-stream-byte-diff-v1
- run-slcs-host-storage-audit-v2
- run-slcs-meiji-read-modes-v1
parents: []
tags:
- slcs
- real-rgb
- meiji
- broadcast
- pseudo-labels
---



## まとめ

Meijiのoutsourceボール注釈と指定Court checkpointから品質重み付き教師を作り、収録ごとのsplitで実RGB SLCSを評価する実験群。2026-09-19時点でMeiji v8の全56clipの観測ファイルを生成し、事後検査で見つかった3cameraのproducer SHA不一致は保存・隔離・再生成により修復した。全168cameraの固定pinsと旧新配列監査が通過し、Court補正を反映したv9の全56clip生成と校正3clipの150完全一致比較も通過した。RGB特徴は全56clipを新たに照合してv9へ再利用済み。一方、v9人物観測の再利用前照合でViTPose SHA不一致が発生し、公開0で停止した。全体教師と全体版SLCSの頑健性は未完成。

教師の主な根拠は[BLCS同条件評価](run-slcs-blcs-meiji-finetuned-eval.md)、[PLCS validation選定重みのtest](run-slcs-plcs-meiji-foot-e60-selected-test.md)、[同一2D観測での実クリップ比較](run-slcs-plcs-meiji-real-final-eval.md)。合成test、観測から作った擬似3Dとの一致度、実画像への再投影を区別する。独立実測3D正解はなく、再投影改善を絶対3D精度と呼ばない。

SLCSの先行試験はMeiji 2クリップとbroadcast 5クリップで、[baseline](run-slcs-rgb-pilot-baseline-selected-conditions-v2.md)と[入力欠損augmentation](run-slcs-rgb-pilot-augmented-selected-conditions-v2.md)を各60epoch比較した。testはbroadcast 1収録のみで、Meiji testは含まれない。augmentationは欠損時の選手精度を改善したがfull入力は微悪化し、ballはtrain/testとも大きな誤差と低分散出力が残った。[train限定の平均位置baselineとの比較](run-slcs-ball-train-mean-v1.md)でも誤差差は小さく、軌道未学習の診断を支持した。現在の重みを頑健な実世界モデルとして採用しない。

全体生成の途中で[人物対応の切替とcache記録不一致](run-slcs-meiji-v6-tracking-qc.md)を確認した。旧v6の生成を取り消し、[時間的な人物対応](run-slcs-meiji-temporal-probe-v1.md)を適用したv7で4clipを生成した。教師の利用率は改善したが、[推論後のchecksum不一致](run-slcs-meiji-checkpoint-hash-probe-v1.md)を再現した。[別processのsha256sumでも不一致](run-slcs-meiji-checkpoint-hash-probe-v3.md)が発生し、実行環境の切り分けを行った。[CPU対照](run-slcs-meiji-checkpoint-hash-cpu-v1.md)と[推論後の命令制限対照](run-slcs-meiji-checkpoint-hash-gpu-control-v1.md)はともに成功し、間欠的な不一致の原因は未確定である。[既知メモリのみの384比較](run-slcs-known-memory-cpu-audit-v1.md)も全件一致し、入力fileを除いた条件では再現しなかった。[同一652frameの反復推論](run-slcs-meiji-inference-repeatability-v1.md)は全33,252要素が完全一致し、8行の二実装hashも一致した。原因解決とは区別し、二重hash・期待checkpoint SHA照合・即時停止を追加した。[肩支持の本番CPU照合](run-slcs-meiji-v8-root-replay-v2.md)を通し、新しいMeiji v8で観測生成を再開した。v7の4clipは比較根拠として保持し、v8の生成完了数とは区別する。

### 次の判断

第3収録の[画像確認](run-slcs-meiji-observation-review-v4.md)までで10clip・267画像・3収録を確認した。観測画像のP0/P1は各視点のnear/farで、教師入力は `associate_people` で共通人物軸へ並べ替える。[productionのCPU照合](run-slcs-meiji-canonical-association-check-v1.md)で全10clipのcam2反転と配列一致を確認した。旧raw-index集計のうちvideo_000/clip_008 P1の2視点腰肩支持は476/479からcanonical軸の447/479へ訂正した。教師生成後の幾何・再投影・速度条件を通す前の診断であり、最終教師coverageではない。

[低支持3clipの追加確認](run-slcs-meiji-observation-review-v5.md)で合計13clip・348画像となった。clip_009 cam2の71frame欠損を特定し、共通人物軸では他視点を含めた2視点腰肩支持が全frameにあることを確認した。追加3clipの支持率99.708–100%も最終教師coverageとは区別する。

[第2・第3収録のCourt画像確認](run-slcs-meiji-court-visual-qc-v1.md)では、保存数値の再計算は一致したが、第3収録cam0の近側baselineに実白線とのずれが見えた。近側baseline上に採用inlierが無く、外挿誤差の可能性を残す。p95は除外点を含む検出残差で独立GT誤差ではない。最終教師の幾何品質と独立した線対応を確認してから採否を判断する。

[白帯の局所画素診断](run-slcs-meiji-baseline-line-audit-v2.md)では第3収録cam0の右側baselineに22.57–26.69pxの上下差があった。既存手動Court注釈は第1収録だけであり、本数値はAIによる画像確認を伴う局所診断として扱う。同checkpointのcrop比較を準備した。[観測全体の事後監査](run-slcs-meiji-v8-observe-v1.md)で3cameraのpeople checkpoint SHA不一致を検出し、後続buildをcancelした。12個の元観測ファイルを保存し、pin照合と再生成を先に進める。[RGB特徴再利用](run-slcs-meiji-v8-feature-reuse-v1.md)は53clipが完了し、元markerが無い3clipが未生成。

[3cameraの再生成と全体監査](run-slcs-meiji-v8-observe-repair-v1.md)では、対象6NPZの全配列が完全一致し、metadata差は既知3fieldだけだった。対象外165cameraのpeople fileは不変、全168cameraの固定pinsも一致した。観測producerの採用検査は通過したが、間欠的な不一致の原因は未確定であり照合と停止規則を維持する。

[Court crop比較](run-slcs-meiji-court-crop-comparison-v1.md)では元6viewのraw/score/Hが完全再現し、Court extentを含むC候補が全viewのfitを通過した。第3収録cam0の局所白帯差は最大26.69→1.51pxへ減り、第1収録の既存注釈への平均誤差も両viewで減った。2passのproduction同値確認と新Meiji v9への分離を進める。

[残3clipの特徴生成後の全体監査](run-slcs-meiji-v8-features-missing-v1.md)は、56clipの特徴schema/設定検証を通過したが、v8 video_001/clip_005/cam2動画の終了時SHAが原本と異なりfailedとなった。同inodeの原本側は前後一致しており、原因は未確定。[別processのsnapshot対照](run-slcs-meiji-video-alias-audit-v1.md)では6digestが期待値と一致し、2snapshotのbyte差は0だった。この限定対照は前回の失敗を取り消さず、原因解決や全体採用完了とは扱わない。

[v9 Courtの通常生成](run-slcs-meiji-v9-court-v1.md)は全56clipで失敗0。校正3clipのinitial/refinedのraw・score・frame index・ROI・Hとcam2旧結果を150項目で完全照合し、110入力の前後SHAも一致した。旧samples所在地を誤った最初の比較driverと失敗記録は保存し、明示rootを修正した。実データやGPU生成の再実行で比較を合わせたものではない。

[v9 RGB再利用](run-slcs-meiji-v9-feature-reuse-v1.md)は全56clip168cameraを公開し、最終1139入力・出力のhashが一致した。[人物観測の再利用](run-slcs-meiji-v9-observation-reuse-v1.md)は142cameraで選択6配列が一致、26cameraに差があったが、公開前2058入力のうちViTPoseだけSHA不一致で停止した。例外後の一致を採用根拠へ置き換えず、公開0のまま読取byteを保存する切り分けへ進む。

[同一読取streamの保存対照](run-slcs-meiji-read-stream-capture-v1.md)はViTPoseの3読取をその場でsnapshotへ保存し、独立processのhashと全byte比較まで一致した。人物選択の実処理を含まない限定対照であり、原因解決とは扱わない。[Windows側の読取監査](run-slcs-host-storage-audit-v1.md)では、WSLのVHDがあるD:のNVMeに4件のresetと、Windowsに1件のbugcheck 0x154を確認した。SHA不一致との因果関係やSSD故障は未確定。実処理中の読取内容を保存する診断を準備する。

[人物観測の読取stream付き再利用](run-slcs-meiji-v9-observation-reuse-v2.md)は初回ViTPose pin照合で停止し、不一致byte列の保存に成功した。2実装の同stream hash、保存snapshotの独立Python2実装と外部sha256sumが全て同じ不一致値となり、長さ・stat・保存chunk照合は一致した。人物選択前で公開0。正常snapshotとは1MiB chunk index1808だけhashが異なり、live fileを再読せず保存内容同士のbyte差分を調べる。

[保存snapshotの全byte比較](run-slcs-meiji-stream-byte-diff-v1.md)で、2,549,075,546bytes中ちょうど1byte/1bitの差を確認した。offset1896501665で0x9D→0xBD、XOR0x20。両snapshotの二実装SHA・stat・長さとmetadata前後照合は通過し、ZIP headerだけで重みarchiveのstored payload内と特定した。原因がstorage/cache/memory/softwareのどこかは未確定で、SSD故障等の断定や失敗結果の採用はしない。

[同位置4KiBの通常/direct read比較](run-slcs-meiji-read-modes-v1.md)はstatx alignmentを確認し、live3readがすべてgood保存blockに一致した。これは限定した時点と位置の結果で、全体生成を再開できる安定性の根拠とはしない。[Windows collectorの実行](run-slcs-host-storage-audit-v2.md)はUNC上の未署名scriptとして実行前に拒否され、新規ログは取得できなかった。ポリシーは変更せず、ローカルコピーでの読み取り収集についてユーザーへ確認している。

ユーザー指定に従い、まずMeiji全体の教師生成・品質確認を完了する。欠落・低支持区間・除外理由を固定し、重みとsource codeを維持する。その後、ball未学習の仮説を1施策ずつ50–70epochで比較し、収録分離した全体版のfull/no_rgb/detector_gap/rgb_only評価へ進む。[固定train窓の項別勾配診断](run-slcs-ball-gradient-probe-v1.md)でmodeによる平滑化項の差と局所勾配を確認したが、損失平滑化過剰説は仮説であり、再学習による施策比較は未実施。

生成の実行方法・採用規則は[実RGBデータ生成ガイド](../../src/tennis_scene/dataset_pipeline/README.md)、出力パス規則は[OUTPUTS.md](../../src/tasks/OUTPUTS.md)を正本とする。中断した学習はfailed nodeとして残し、再開run・validation選定・独立test評価を別nodeへ分離している。
