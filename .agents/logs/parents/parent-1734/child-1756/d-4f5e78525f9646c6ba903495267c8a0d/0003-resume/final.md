正式レビューのmajor指摘を修正しました。PRはReadyで、最終HEADは `68aebf4f5d404f6e57d2d9cb5380d8ce679c9aa9` です。 [PR #2271](https://github.com/Motoki0705/tennis-lab/pull/2271)

- 実際のデコードと前処理の条件を共通関数で取得し、外観特徴・人物対応のキャッシュと完成マーカーに反映しました。条件が変わると旧成果物が拒否されることをCPU統合テストで確認しました。
- #2270との競合を解消し、MDD検査を修正した #2280 を取り込みました。最終HEADでCPU回帰1,695件、MDD境界テスト2件、pre-commit、本体CI全14ジョブ、Knowledge CI全2ジョブが成功しました。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44473619385) [Knowledge CI](https://github.com/Motoki0705/tennis-lab/actions/runs/44473618928)
- 保存済み3clipのCPU再生では816件すべてを再利用し、前回の結果と一致しました。#1757で必要な旧キャッシュの再生成手順はPR本文に記載しました。

worktreeに未コミット変更はなく、作業プロセスやGPUキューのジョブも残っていません。