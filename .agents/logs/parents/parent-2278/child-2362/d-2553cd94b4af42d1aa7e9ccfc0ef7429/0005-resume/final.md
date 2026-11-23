設定の収集エラーとGVHMRの`forward`境界違反を修正し、PR #2379の同じブランチに `9e3c024e9` をpushしました。既定値は `pipeline.yaml` で合成し、GVHMRのlength・maskはFP32復号の直前に検証します。TensorRTのexport・buildも設定監査に登録しました。 [ドラフトPR #2379](https://github.com/Motoki0705/tennis-lab/pull/2379)

設定・構造検査は419件、CPU回帰は2,242件が成功し、既定FP32のpublication identityと38コンポーネントの設定が導入前と一致しました。main取り込み後は #2371 の対象テストとknowledge検証が通過し、knowledgeの80番と81番を保持しました。pre-commitも通過しています。

GPU実測は既存の記録を参照します。DINOは単体基準未達で、採用は #2363 の全クリップE2Eで判断します。新HEADのPython CI成功後にドラフトを解除します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2379,"head":"9e3c024e961df48c2596711d80e7c21aec79b804","resume_on_success":true,"next_step":"ドラフト解除と最終報告"}}