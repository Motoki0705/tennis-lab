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
parents: []
tags:
- slcs
- real-rgb
- meiji
- broadcast
- pseudo-labels
---



## まとめ

Meijiのoutsourceボール注釈と指定Court checkpointから品質重み付き教師を作り、収録ごとのsplitで実RGB SLCSを評価する実験群。2026-09-18時点ではMeiji全体の生成・品質確認を進行中で、全体版SLCSの頑健性はまだ確認していない。

教師の主な根拠は[BLCS同条件評価](run-slcs-blcs-meiji-finetuned-eval.md)、[PLCS validation選定重みのtest](run-slcs-plcs-meiji-foot-e60-selected-test.md)、[同一2D観測での実クリップ比較](run-slcs-plcs-meiji-real-final-eval.md)。合成test、観測から作った擬似3Dとの一致度、実画像への再投影を区別する。独立実測3D正解はなく、再投影改善を絶対3D精度と呼ばない。

SLCSの先行試験はMeiji 2クリップとbroadcast 5クリップで、[baseline](run-slcs-rgb-pilot-baseline-selected-conditions-v2.md)と[入力欠損augmentation](run-slcs-rgb-pilot-augmented-selected-conditions-v2.md)を各60epoch比較した。testはbroadcast 1収録のみで、Meiji testは含まれない。augmentationは欠損時の選手精度を改善したがfull入力は微悪化し、ballはtrain/testとも大きな誤差と低分散出力が残った。[train限定の平均位置baselineとの比較](run-slcs-ball-train-mean-v1.md)でも誤差差は小さく、軌道未学習の診断を支持した。現在の重みを頑健な実世界モデルとして採用しない。

全体生成の途中で[人物対応の切替とcache記録不一致](run-slcs-meiji-v6-tracking-qc.md)を確認した。旧v6の生成を取り消し、[時間的な人物対応](run-slcs-meiji-temporal-probe-v1.md)を適用したv7で4clipを生成した。教師の利用率は改善したが、[推論後のchecksum不一致](run-slcs-meiji-checkpoint-hash-probe-v1.md)を再現した。[別processのsha256sumでも不一致](run-slcs-meiji-checkpoint-hash-probe-v3.md)が発生し、実行環境の切り分けを行った。[CPU対照](run-slcs-meiji-checkpoint-hash-cpu-v1.md)と[推論後の命令制限対照](run-slcs-meiji-checkpoint-hash-gpu-control-v1.md)はともに成功し、間欠的な不一致の原因は未確定である。[既知メモリのみの384比較](run-slcs-known-memory-cpu-audit-v1.md)も全件一致し、入力fileを除いた条件では再現しなかった。[同一652frameの反復推論](run-slcs-meiji-inference-repeatability-v1.md)は全33,252要素が完全一致し、8行の二実装hashも一致した。原因解決とは区別し、二重hash・期待checkpoint SHA照合・即時停止を追加した。[肩支持の本番CPU照合](run-slcs-meiji-v8-root-replay-v2.md)を通し、新しいMeiji v8で観測生成を再開した。v7の4clipは比較根拠として保持し、v8の生成完了数とは区別する。

### 次の判断

第3収録の[画像確認](run-slcs-meiji-observation-review-v4.md)までで10clip・267画像・3収録を確認した。観測画像のP0/P1は各視点のnear/farで、教師入力は `associate_people` で共通人物軸へ並べ替える。[productionのCPU照合](run-slcs-meiji-canonical-association-check-v1.md)で全10clipのcam2反転と配列一致を確認した。旧raw-index集計のうちvideo_000/clip_008 P1の2視点腰肩支持は476/479からcanonical軸の447/479へ訂正した。教師生成後の幾何・再投影・速度条件を通す前の診断であり、最終教師coverageではない。

[低支持3clipの追加確認](run-slcs-meiji-observation-review-v5.md)で合計13clip・348画像となった。clip_009 cam2の71frame欠損を特定し、共通人物軸では他視点を含めた2視点腰肩支持が全frameにあることを確認した。追加3clipの支持率99.708–100%も最終教師coverageとは区別する。

[第2・第3収録のCourt画像確認](run-slcs-meiji-court-visual-qc-v1.md)では、保存数値の再計算は一致したが、第3収録cam0の近側baselineに実白線とのずれが見えた。近側baseline上に採用inlierが無く、外挿誤差の可能性を残す。p95は除外点を含む検出残差で独立GT誤差ではない。最終教師の幾何品質と独立した線対応を確認してから採否を判断する。

ユーザー指定に従い、まずMeiji全体の教師生成・品質確認を完了する。欠落・低支持区間・除外理由を固定し、重みとsource codeを維持する。その後、ball未学習の仮説を1施策ずつ50–70epochで比較し、収録分離した全体版のfull/no_rgb/detector_gap/rgb_only評価へ進む。[固定train窓の項別勾配診断](run-slcs-ball-gradient-probe-v1.md)でmodeによる平滑化項の差と局所勾配を確認したが、損失平滑化過剰説は仮説であり、再学習による施策比較は未実施。

生成の実行方法・採用規則は[実RGBデータ生成ガイド](../../src/tennis_scene/dataset_pipeline/README.md)、出力パス規則は[OUTPUTS.md](../../src/tasks/OUTPUTS.md)を正本とする。中断した学習はfailed nodeとして残し、再開run・validation選定・独立test評価を別nodeへ分離している。
