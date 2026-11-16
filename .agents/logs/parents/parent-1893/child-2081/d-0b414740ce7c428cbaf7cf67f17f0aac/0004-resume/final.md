[PR #2085](https://github.com/Motoki0705/tennis-lab/pull/2085) を更新・通常pushしました。

- branch: `codex/git-sync-interactive-cli-guard`
- HEAD: `2ced17f64`
- main `0ff897fc5` を取り込み、7衝突を解消。16入口の薄さ、lock順、#2116/#2131の読取・鮮度契約を維持しました。
- CIで見つかった重複SIGTERMによる停止記録欠落も、mainで再現し別commitで修正しました。
- 重点78件・関連1,067件成功、停止修正後397件成功。pre-commitも成功。
- [CI #4272](https://github.com/Motoki0705/tennis-lab/actions/runs/43711192655)：全14ジョブ成功、24,245成功・101 skip。Pythonの対象HEAD一致も確認済みです。

実ホストは未操作です。追加修正を含む親の再レビュー・mergeと、旧版CLI終了確認、git-sync ON・status/logs確認が残ります。