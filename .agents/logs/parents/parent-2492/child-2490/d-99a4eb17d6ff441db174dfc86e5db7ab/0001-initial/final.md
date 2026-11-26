共通adapter仕様を [#2490](https://github.com/Motoki0705/tennis-lab/issues/2490#issuecomment-7129915413) に掲載し、ViTPose-Hを固定疑似BBoxで採点するCLIを [draft PR #2500](https://github.com/Motoki0705/tennis-lab/pull/2500) に追加しました。PRは一時的に #2498 のbranchをbaseとし、#2497も取り込んでいます。

#2495の公開manifestと元動画24本のSHA-256を照合し、6,844画像・26,276人物行と各動画の先頭評価フレームを確認しました。pytestは新規4件を含む29件が成功し、pre-commitとknowledge検査も通過しました。

CIが実行中です。残件はViTPose-HのGPU採点、#2496の速度基準と候補6モデルの結果の統合、難条件別集計、同期可視化、Pareto比較です。

`codex/parent-2492-issue-2490-primary` · `7f7a84ba619be8e06a972f38a491f0176bb18e7d`

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2500,"head":"7f7a84ba619be8e06a972f38a491f0176bb18e7d","resume_on_success":true,"next_step":"固定疑似BBoxでViTPose-Hのflipあり・なしの精度採点を共有GPU queueへ登録する。TRT16はflip設定とprofileが一致するengineを照合する。速度には #2496 の基準値を使い、補助担当の成果を同じPRへ統合する。"}}