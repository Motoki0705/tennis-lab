main `f53d0fd5b029377aca7c3e6d4c8af90d26a01ab6` を取り込み、`codex/parent-1739-issue-2444-primary` をpushしました。完全HEADは `907e4d66b3f78a3db2d15a4da5733929ae5e60b1` です。GitHubで競合なしを確認し、`Closes #2444` を維持してPR本文を更新しました。 [PR #2449](https://github.com/Motoki0705/tennis-lab/pull/2449)

- `source_identity.py` と `inventory.py` を統合し、実ソース12ファイルのハッシュを照合しました。16通りの設定でmainと同じキャッシュ識別になることを確認しました。
- `test_fp32_cache_compatibility.py` にはmainの保存fixtureを採用しました。FastHMRのテストは、#2444の互換変換を適用した後にECPose・FastHMR・単眼PLCSの識別を検証する形に修正しました。
- knowledgeの3ノードを98〜100に割り当て、`.sequence` を100にしました。

再試験は **1,669件すべて成功（367.75秒）**。pre-commitのruff・mypyも成功し、knowledge検査はエラー・警告ともに0件でした。 [実行中のCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45778015602)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2449,"head":"907e4d66b3f78a3db2d15a4da5733929ae5e60b1","resume_on_success":false}}