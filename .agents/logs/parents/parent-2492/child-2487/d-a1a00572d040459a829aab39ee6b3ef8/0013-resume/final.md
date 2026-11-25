EdgeCrafter PT16 の6重み×batch 1・2、全12条件を検証しました。固定96画像で各300反復の入力順と集計が一致し、統合側の比較コードでも新しい batch 1 の6条件を読み込めました。[計測記録・原本SHA-256](https://github.com/Motoki0705/tennis-lab/blob/e9a12d172279a35b5885bbdd37e651772978aac8/knowledge/runs/run-meiji-player-model-comparison/edgecrafter-pt16-speed-matched.json)

- **ECDet-X O365** の batch 1 wall P50 は **50.75 ms/画像**、DINO は165.69 ms/画像でした。疑似GTの Recall@0.50／0.75 は97.97%[local path]、DINO は96.69%[local path]。
- **ECPose L/X** は **56.89／56.06 ms/画像**、DINO＋ViTPose-H は266.30 ms/画像でした。欠測込みOKSは0.291／0.259で、基準は0.763でした。

batch 2 の精度は未評価です。旧 TRT16 の速度は初回のスレッド設定で得た暫定値で、精度低下の原因は未特定です。

`codex/parent-2492-issue-2487-edgecrafter` の `e9a12d172279a35b5885bbdd37e651772978aac8` を push しました。pytest 89件、pre-commit、ナレッジ検証が通過しました。[#2487の記録](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7164777819) · [起動したCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45630520651)

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}