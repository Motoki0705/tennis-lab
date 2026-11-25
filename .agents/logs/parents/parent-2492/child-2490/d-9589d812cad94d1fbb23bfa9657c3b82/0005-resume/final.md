PMPose-L/Hの全6条件を正準照合で再採点しました。Recall・OKS・PCKを含む全数値が一致し、同一IoUの解決順を示すメタデータが追加されました。PT32の平均OKSはLが **0.77663**、Hが **0.78084** です。 [精度記録](https://github.com/Motoki0705/tennis-lab/blob/317e05e1b5808a5058f3fc47c990c16a62966bec/knowledge/runs/run-meiji-topdown-pose-comparison/pmpose_accuracy.json)

PT32に対する座標差は全条件でP95が0.032 px未満でした。最大差1,926.6 pxは疑似GTの採点対象外の関節で発生し、採点対象内でも最大116.2 pxの差がありました。該当フレームと補助スコアの診断を保存しました。 [診断記録](https://github.com/Motoki0705/tennis-lab/blob/317e05e1b5808a5058f3fc47c990c16a62966bec/knowledge/runs/run-meiji-topdown-pose-comparison/PMPose.md%23auxiliary-scores)

ブランチ `codex/parent-2492-issue-2490-pmpose`、HEAD `317e05e1b5808a5058f3fc47c990c16a62966bec`。57テスト成功（11.37秒）。速度・VRAM計測に必要な#2497の修正版は公開待ちです。公開後にこの担当を再開してください。 [#2497](https://github.com/Motoki0705/tennis-lab/pull/2497)

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}