[PR #1890](https://github.com/Motoki0705/tennis-lab/pull/1890)で `pr-1890-1` を修正しました。

- branch: `codex/decision-log-acceptance`
- HEAD: `89701da79`
- 復号・整形・切詰め後の表示値もHTML escape前にGuardで検査し、違反は固定エラーで拒否します。
- 修正前8件失敗を確認。追加回帰14ケースを含む対象215ケース、pre-commitが成功しました。
- mainの205記録も拒否なく再生成できました。58 README、最大29,310 bytes、event変更ゼロ、公開policy合格です。
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42947632930)：Python・Lint・CPU全10分割が成功しました。

指摘の解消確認・merge・新表示の実公開とGitHub表示確認は親へ残しています。