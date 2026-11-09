実装を完了し、[PR #1761](https://github.com/Motoki0705/tennis-lab/pull/1761)を作成しました。

- branch: `codex/v3-active-ball-publication`
- commit: `bb96f4206`
- v3の推定点を観測と同じ重み1でsnapshot・学習readerへ接続。unknownのmask、収録group/split、冪等な公開・中断復旧を検証しました。
- 周辺回帰486件、最終統合14件、pre-commit成功。
- 対象HEADのCIはLint・CPU全10分割・`Python`すべて成功。worktreeはcleanです。

旧汎用storeとの混在は明示拒否します。CLI配線・旧データ移行・実動画受入は後続範囲で、学習は起動していません。追加判断の依頼はなく、統合レビュー・merge・merge後mainのCI確認を親へ引き継ぎます。