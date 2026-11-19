[PR #2085](https://github.com/Motoki0705/tennis-lab/pull/2085) を修正・push済みです。

- branch: `codex/git-sync-interactive-cli-guard`
- HEAD: `54e9c5ce3`
- 16入口を薄いbootstrapに変更し、checkout → registryの取得順へ統一。保守status/stopも維持しました。
- 追加36ケース・故障注入・関連テスト・pre-commitを検証済み。
- [CI #4240](https://github.com/Motoki0705/tennis-lab/actions/runs/43679017618)：全14ジョブ成功、17,306成功・91 skip。

途中のCI失敗とfixture修正、ローカル環境の制約はPR・子issueに記録しました。

実ホストは未操作です。親による指摘解消確認・merge、旧版CLI終了確認、git-sync ONとstatus/logs確認が残ります。