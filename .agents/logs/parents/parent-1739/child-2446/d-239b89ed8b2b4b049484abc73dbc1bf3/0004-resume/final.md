meiji_4camのGPUジョブが完了し、新経路は全69本のSceneResultを保存しました。完了表では、各側2人のGVHMRと有効な身体配置、ボール3Dがそろったのは旧経路52/69本、新経路69/69本です。

- 4人の身体配置が同時に有効なフレームは、旧29,878/46,228（64.63%）、新41,949/46,228（90.74%）。新経路の最低は `clip_016` の198/1,214です。
- ボール3Dは両経路とも45,958/46,228フレームで有効でした。欠測270フレームはすべて `clip_064` です。
- Meiji 3camの未見3本では、IDF1は旧92.07%・新80.42%、選別recallは旧85.30%・新67.25%でした。

両経路の69本の完了表をknowledgeに記録し、段階2の身体表示を `clip_009` で試写しました。全69本の読戻し・球配列の照合と、比較ページ16本の再生成をCPUジョブに渡しました。結果を受けて記録を確定します。

ブランチ `codex/parent-1739-issue-2446-primary`、HEAD `8328bfceec697649eb537d575e69f2e9fa3be637`。[Draft PR #2470](https://github.com/Motoki0705/tennis-lab/pull/2470)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261124T100319Z_0e179f3398ad_i2446-verify-and-render"]}}