Project I/Oとboard表示・Codex sessions探索を[PR #2084](https://github.com/Motoki0705/tennis-lab/pull/2084)へ統合し、題名・本文を更新してpushしました。

- branch: `codex/board-project-io`
- HEAD: `8faefc07a`
- 関連テスト: **732 passed / 1 skipped**（ローカルのPlaywright未設定）
- ruff・mypy・pre-commit成功
- 対象HEADの[Python CI成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43660260098/job/132657623351)
- 作業treeはclean、取込済みの別branchは削除済み

初回CIで既存の削除テスト2件が失敗しましたが、同じHEADの再実行で成功しました。根本原因は未確定としてPRに記録しています。

配送fenceは未実装です。残作業は親側の統合レビュー・merge・merge後main CI確認です。