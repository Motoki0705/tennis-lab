main `2736a24a3` を取り込み、5ファイルの衝突を解消しました。

- branch: `codex/git-sync-interactive-cli-guard`
- commit: `de02aadf2`（通常push済み）
- [PR #2085](https://github.com/Motoki0705/tennis-lab/pull/2085)：本文更新済み、merge可能
- 全16入口の薄いbootstrapと#2149の変更を保持。取得順を `checkout → ownership → registry` に統一
- 関連699件・pre-commit成功。追加テストの修正前失敗も確認
- 対象HEADの[CI #4319](https://github.com/Motoki0705/tennis-lab/actions/runs/43815673956)：全14ジョブ成功、24,598成功・101 skip、延期0

[子issueへ証拠を記録済み](https://github.com/Motoki0705/tennis-lab/issues/2081#issuecomment-6909463438)。実ホストは未変更です。残作業は親による再レビュー・merge、旧CLI終了確認後のgit-sync ONと実ホスト確認です。