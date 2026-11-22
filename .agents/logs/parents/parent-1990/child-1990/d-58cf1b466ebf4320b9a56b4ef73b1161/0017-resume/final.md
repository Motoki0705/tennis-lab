[PR #2143](https://github.com/Motoki0705/tennis-lab/pull/2143) をレビュー可能にしました。`Refs #1990` を維持しています。  
branch: `codex/storage-resident-data`[local path]`f889cde7c`

- 常駐6単位は、既存のSHA-256照合・復旧経路を再利用し、削除対象から除外しました。未管理のE:参照は9パスを明示し、誤ったリンク先・未mountは拒否します。
- 新規テスト37件。修正前コードで24件の失敗を確認し、最終検証は**6,666件成功、pre-commit成功**。対象HEADの[CIもPythonを含む全14ジョブ成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43719839189)です。
- 実機の読取専用QAは最終unknown 0。metadata・rootリンクの時刻は不変、既存リンクのtarget読取りは0回でした。
- [更新した手順書](https://github.com/Motoki0705/tennis-lab/blob/f889cde7cabc4dc03a9b79c5ad9dc037ec663142/tmp/storage-tier-phase2-runbook.md)に、照合コピー、9本のリンク作成、空outputs、Clip Studio停止・再開、worktree CLI、最後にまとめるsudo手順を反映しました。

CI最長は32分28秒でした。新規37件を含まない既存shardの実測と偏りをPR本文に記録しています。

実機の配置切替・実index・hydrate・削除・unit・sudoは未実施です。追加の配置判断は不要ですが、正式レビュー・mergeと、timer有効化前のmain側コード[local path]。