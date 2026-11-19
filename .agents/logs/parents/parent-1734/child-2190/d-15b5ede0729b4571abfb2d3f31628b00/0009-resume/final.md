[PR #2200](https://github.com/Motoki0705/tennis-lab/pull/2200)に「cacheへの影響」を追記しました。

- 旧court cacheは3 cameraとも再計算判定。loadはidentity不一致の`ValueError`で停止。
- `src/**[local path]`のfingerprintが全componentへ波及する既存の仕組みを確認。
- CPUのcache/loadテスト：11件成功、4.74秒。

無効化の範囲・再計算の必要性・意図を明記しました。本文のみ更新し、branch `codex/court-acceptance-postprocess`[local path]`9507c5882`と成功済みCIを維持しています。GPU実行・merge・closeは行っていません。