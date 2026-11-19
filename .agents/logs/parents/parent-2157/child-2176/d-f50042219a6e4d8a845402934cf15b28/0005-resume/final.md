`origin/main`（`569ecad37`）をmergeし、14ファイルの競合を解消してpushしました。rebase・force pushはしていません。

- branch: `codex/merge-pipeline-switch`
- 新HEAD: `6d16be39d153f8a8a579a2203311cd8be28a4b7a`
- [PR #2184](https://github.com/Motoki0705/tennis-lab/pull/2184): **mergeable=true**

#2085のCLI保護・lock取得順・停止処理を新しい自動進行へ移しました。旧レビュー操作・厳密照合は復活させていません。#2192のquarantine復旧実装とテストはmainの内容を維持しています。[14ファイルそれぞれの解消方針と検証記録](https://github.com/Motoki0705/tennis-lab/pull/2184#issuecomment-6911492229)をPRに掲載しました。

個人環境変数を除いた関連テストは、重複なしで**1,163ケース成功**。名前衝突の最終4ケース、pre-commit、ruff、主要運用コード10ファイルのmypyも成功しました。

[新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43832339977)は開始済みで、完了未確認です。追加判断はありません。統合レビュー・merge・実PRでの完了条件12の確認は親へ引き継ぎます。validatorは指定なしで0回です。