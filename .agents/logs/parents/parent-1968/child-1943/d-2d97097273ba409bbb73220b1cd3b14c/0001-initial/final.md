## 結果

受入条件7項目を満たしました。

- 最新runのJSON返却：初回・resume・planningのE2E成功。
- コマンド行の自己一致回帰：終了後に復帰することを確認。
- 終了済みの即時返却・run未登録の拒否：確認済み。
- runner喪失：待機前・待機中のSIGKILLを生存不明として返却。
- ロック非保持：待機中に同じ親の`note`更新が成功。
- timeout：status応答停止時も124で終了し、workerを維持。
- README・skill：背景待機の手順をREADMEへ集約。

追加29件・既存161件が成功。ruff・mypy・pre-commitも成功しました。既存起動処理を保つため、`setsid`＋`wait-worker`を採用しています。

実agent新規起動・OS再起動は未実施です。validatorは指定なし、試行／完了0／0回です。

## branch・commit・PR・CI

- branch：`codex/parent-wait-worker`
- commit：`37b5caca637cae0a2e3336c8b2b2953c27859d4a`
- PR：[#1975](https://github.com/Motoki0705/tennis-lab/pull/1975)
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43386837283)：`Python`を含む全14 job成功。
- 作業treeはcleanです。

## issueへの報告案

なし。受入証拠・方式の理由・実測はPR本文に記載済みです。

## 親への判断依頼

なし。統合レビュー・merge・merge後mainのCI確認を親へ引き継ぎます。