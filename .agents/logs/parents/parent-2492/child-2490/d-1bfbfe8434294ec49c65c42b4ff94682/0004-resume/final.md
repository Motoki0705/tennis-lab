RTMPose-M/Xの6条件を、保存済み予測各26,276件から #2498 のマージ済み採点器で再採点しました。OKS・PCK・関節誤差・欠測数は全条件で一致しました。固定BBoxの照合は全条件26,276/26,276件で、Recall@0.50・@0.75ともに100%でした。新しい照合順序のメタデータと再採点ファイルのハッシュを保存しました。[再採点結果](https://github.com/Motoki0705/tennis-lab/blob/9323512c9725be727bf02711bccde21dd5e11fa8/knowledge/runs/run-meiji-topdown-pose-comparison/rtmpose_accuracy.json)

統合用 branch は `codex/parent-2492-issue-2490-rtmpose`、commit は `9323512c9725be727bf02711bccde21dd5e11fa8` です。90テスト、再採点コードのpre-commit、knowledge検証が通過しました。

速度job 2件は親の指示どおり実行前に取り消されており、速度・VRAMの実測値はありません。再開は #2497 修正版と親からの再投入指示待ちです。

PARENT_RESULT {"schema_version":1,"state":"needs_parent","wait":null}