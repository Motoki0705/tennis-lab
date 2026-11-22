必須の棚卸し・移行設計・分割案をまとめました。

- [調査報告・保存先の比較](https://github.com/Motoki0705/tennis-lab/blob/2ac198a07793475bd892b4d56d9140ffd8ec83e5/tmp/1881-plan/plan.md)
- [イベント別の投稿元・読取元・移行先の対応表](https://github.com/Motoki0705/tennis-lab/blob/2ac198a07793475bd892b4d56d9140ffd8ec83e5/tmp/1881-plan/inventory.md)
- [件数・指定実例・冪等性の設計](https://github.com/Motoki0705/tennis-lab/blob/2ac198a07793475bd892b4d56d9140ffd8ec83e5/tmp/1881-plan/counting-and-cases.md)
- [既存テストと追加検証の計画](https://github.com/Motoki0705/tennis-lab/blob/2ac198a07793475bd892b4d56d9140ffd8ec83e5/tmp/1881-plan/test-plan.md)

推奨は、機械証拠を対象baseを進めない専用Git refへ保存し、認証publisherの公開receiptで照合する方式です。既存 `decision_logs` は判断・閲覧の導線として再利用します。現在の補助ログをそのまま権限証拠にすることや、固定レビュー要求をmainへ同期公開することには問題があります。代替の「証拠PRコメント正本＋後日Git保存」も比較しました。

旧コメントの読取互換は維持し、新証拠の欠落時には旧成功へfallbackしない設計を提案します。実装は「保存・読取互換」→「全投稿経路とgateの切替」の2PRを推奨します。**保存方式と分割が親の判断点です。**

6件の全320コメントを取得・件数照合済みです。元の308件と#1796の追加12件を分離し、指定実例13件を確認しました。**全件の意味分類と適用後の総投稿数は未確定**です。残りを分類できる原文資料・台帳・集計手順を保存しています。

現行契約の既存テストは **16 passed／48.70秒**。新方式は未実装・未検証です。branchは `codex/parent-comment-curation`、HEADは `2ac198a07` のまま。コード・GitHubへの書込、コミット、PR作成はありません。validatorは未指定のため0回です。