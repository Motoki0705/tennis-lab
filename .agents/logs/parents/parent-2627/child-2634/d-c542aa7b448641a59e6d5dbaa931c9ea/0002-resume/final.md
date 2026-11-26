**第6章の主張を測定範囲で支持します。** 同じ検出・追跡入力で画像2Dから現行のコート座標選別へ替えると、Meiji 3cam予約3本で選手precisionは **63.25→100%**、recallは **99.17→99.29%**、group IDF1は **74.98→99.71%**。既知の非選手の誤採用は **12,690→0件**でした。開発4本でもprecisionは85.55→99.98%に上がり、recallは95.37→95.32%に下がりました。

予約3本のpair F1は77.35→99.43%でした。両条件で対応が成立した2本では99.42→99.30%で、全件での改善は主に対応成立が2本から3本に増えたためです。

Meiji 4camでは画像2Dの全69本がコート向き判定で停止しました。コート滞在をraw ID単位で測る条件は52本、現行は55本で三角測量まで完了し、現行の3D選手フレーム被覆は **72.41%** でした。4camの独立した選手ID精度、絶対3D誤差、向きを固定した場合の差は未測定です。[全結果と再現手順](https://github.com/Motoki0705/tennis-lab/blob/edbbcf11e6311ed88d5b5f29ca6af4e7c09dfd78/knowledge/runs/run-i2634-player-selection-ablation/README.md)

入力ハッシュ、全76本×4条件の結果、先行run 10・11の13条件を[記録専用PR #2660](https://github.com/Motoki0705/tennis-lab/pull/2660)にまとめました。現行条件は保存済み選別と276視点で一致し、完了・停止理由と3D関節有効マスクは69本で一致しました。pytest 34件とナレッジ検査は通過し、CI待ちです。

branch `codex/parent-2627-issue-2634-primary`  
HEAD `edbbcf11e6311ed88d5b5f29ca6af4e7c09dfd78`

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2660,"head":"edbbcf11e6311ed88d5b5f29ca6af4e7c09dfd78","resume_on_success":false}}