`origin/main` をマージし、4件の競合を解消しました。ローカルの完全HEADは `157d618337108568d36d29931695c4671ed6e6a2` です。 [PR #2449](https://github.com/Motoki0705/tennis-lab/pull/2449)

- `source_identity.py` と `inventory.py` を統合し、mainのrecipeとボールイベントCLIに #2444 の登録を加えました。実ファイル12件のハッシュを照合し、16通りの設定でmainと同じキャッシュ識別になることを確認しました。
- FP32の保存fixtureはmainの版を採用し、既存テストで実ソースと `inventory.py` の再変更時の挙動を確認しました。1件成功（6.01秒）です。
- #2444のknowledgeノード3件を98〜100へ移し、`.sequence` を100にしました。knowledge検査はエラー・警告ともに0件、pre-commitのruff・mypyも成功しました。

関連pytestをsystemdのローカルジョブで実行中です。完了結果を受け取ったターンで確認し、成功後にpushとPR本文の更新を行います。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T072946Z_ab757898a5ed_i2444-merge-cpu"]}}