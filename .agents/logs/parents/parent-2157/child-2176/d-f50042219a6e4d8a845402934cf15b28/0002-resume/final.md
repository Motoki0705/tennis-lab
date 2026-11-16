[PR #2184](https://github.com/Motoki0705/tennis-lab/pull/2184)へ修正をpushしました。

- branch: `codex/merge-pipeline-switch`
- HEAD: `4017f9a1c9056578d54bd60c0fd70ef9af0db1d0`

5件とも修正し、回帰テストが通過しました。

1. workerなしでも対象Issueの停止を尊重。独立PRの継続と保存段階での停止を検証。
2. 実ProgressionGitHub経由でdispatcher修正を起動。全体一覧を取得せず、終了報告の消費まで検証。
3. 手動・dispatcher双方で同じworkerのmajor修正後に明示続行。別sessionは拒否。
4. `merge-confirm`で旧方式のmerge済み成果を後確認。親の完了確認まで検証。
5. 証拠用Issueと依存PRをレビュー資料・MCPへ追加。取得可能範囲と範囲外拒否を検証。

追加は10関数・28ケース。旧テスト対応表も訂正しました。[修正報告と証拠](https://github.com/Motoki0705/tennis-lab/pull/2184#issuecomment-6908620742)をPRへ記録済みです。

個人環境変数を外したe2e全体は **4,137成功・2失敗（27分16秒）**。元のCI失敗5ファイルは173ケースすべて成功しました。残る2件は未変更mainでも再現する環境問題（pidfd API欠落、外部`tests`パッケージとの衝突）です。pre-commit・変更コードのruff/mypyも成功しています。

[新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43813238309)は開始済み、完了未確認です。validatorは指定なしで0回。統合レビュー・merge・merge後の実PR確認は親へ引き継ぎます。