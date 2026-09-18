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

SLCSの先行試験はMeiji 2クリップとbroadcast 5クリップで、[baseline](run-slcs-rgb-pilot-baseline-selected-conditions-v2.md)と[入力欠損augmentation](run-slcs-rgb-pilot-augmented-selected-conditions-v2.md)を各60epoch比較した。testはbroadcast 1収録のみで、Meiji testは含まれない。augmentationは欠損時の選手精度を改善したがfull入力は微悪化し、ballはtrain/testとも大きな誤差と低分散出力が残った。現在の重みを頑健な実世界モデルとして採用しない。

全体生成の途中で[人物対応の切替とcache記録不一致](run-slcs-meiji-v6-tracking-qc.md)を確認した。旧v6の生成を取り消し、[時間的な人物対応](run-slcs-meiji-temporal-probe-v1.md)を適用したv7で4clipを生成した。教師の利用率は改善したが、[推論後のchecksum不一致](run-slcs-meiji-checkpoint-hash-probe-v1.md)を再現し、全体生成前に原因を切り分けている。

### 次の判断

ユーザー指定に従い、まずMeiji全体の教師生成・品質確認を完了する。欠落・低支持区間・除外理由を固定し、重みとsource codeを維持する。その後、ball未学習の仮説を1施策ずつ50–70epochで比較し、収録分離した全体版のfull/no_rgb/detector_gap/rgb_only評価へ進む。現時点の損失平滑化過剰説は仮説であり、施策比較による検証は未実施。

生成の実行方法・採用規則は[実RGBデータ生成ガイド](../../src/tennis_scene/dataset_pipeline/README.md)、出力パス規則は[OUTPUTS.md](../../src/tasks/OUTPUTS.md)を正本とする。中断した学習はfailed nodeとして残し、再開run・validation選定・独立test評価を別nodeへ分離している。
