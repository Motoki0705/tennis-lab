#2462 の `497195ff38ac8bbe49baefac64cffddd64515ba7` を #2470 の作業ツリーに取り込みました。`scene_index.py` とFP32回帰テストの競合を解消し、knowledgeは #2462 の `.sequence` 105 と既存run IDを採用しました。

- #2450 → #2462 → #2470 の順で合成したGitツリーを照合しました。外部ボールCLIの登録を保持し、既存recipeのキャッシュ識別が全16条件で一致すること、対象20ファイルはいずれも編集すると互換条件が失効することを確認しました。
- `publish` と `restore` を #2462 の共通切替処理に接続しました。片方の更新で停止すると、SLCS、scene参照解決、比較CLI、完了表が同じ復旧記録を示しました。公開の再実行と復元後の読戻し、保存された旧世代の参照も確認しました。

重点テスト7件が8.66秒で成功しました。関連CPUテスト、pre-commit、knowledge検査をローカルジョブに登録しました。結果確認後にコミット、push、PR本文の更新へ進みます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T145128Z_d893534e97_i2446-review-r1-123-check"]}}