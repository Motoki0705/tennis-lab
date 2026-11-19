[PR #2153](https://github.com/Motoki0705/tennis-lab/pull/2153)の修正を完了しました。

branch: `codex/dino-batch-api`  
新HEAD: `bf927e61d3d7a6fb07dbba6d7dc31e6679a306f6`

公開24ファイルからsession IDと環境固有パスを除去し、残存0件を確認しました。原本22件はGit外へ保全し、hash・数値結果・連番42を維持しています。

CPU再現準備、pre-commit、knowledge検証が成功。[Pythonを含むCI全14ジョブ](https://github.com/Motoki0705/tennis-lab/actions/runs/43734155406)と[Knowledge全チェック](https://github.com/Motoki0705/tennis-lab/actions/runs/43734155089)も成功しました。

作業ツリーはcleanです。統合レビュー・merge・merge後main CIを親へ引き継ぎます。