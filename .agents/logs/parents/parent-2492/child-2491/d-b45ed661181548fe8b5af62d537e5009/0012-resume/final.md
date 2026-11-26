FastHMRへの置き換え見送りを、速度・目視資料・初回試行の記録とsummaryに反映しました。目視評価は未実施です。標準recipeはGVHMRを使い、FastHMRは明示選択の経路として残ります。[記録PR #2585](https://github.com/Motoki0705/tennis-lab/pull/2585)

現行TRT16のwall P50は、保存CSVから換算して **25.54 ms/人・フレーム** と記載しました。親の記録の25.55 msとの差は0.01 msで、見送り判断に影響しません。

`codex/parent-2492-issue-2491-decision` のHEAD `d7ffb585250e69ba492681c99735617da2e25287` をpushしました。ノード747件と再現パス577参照の検証が通り、`Closes #2491` と実行証拠をPR本文に記載しました。[Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45731108511)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2585,"head":"d7ffb585250e69ba492681c99735617da2e25287","resume_on_success":false}}