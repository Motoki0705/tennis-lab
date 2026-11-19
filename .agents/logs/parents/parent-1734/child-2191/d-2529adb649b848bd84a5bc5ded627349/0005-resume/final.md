[PR #2222](https://github.com/Motoki0705/tennis-lab/pull/2222) に「既定clipモードのcacheへの影響」を追加しました。

数値不変とcache非互換を区別し、再計算範囲、load拒否、annotation.jsonのstale化と明示overwrite、既存の全src fingerprintによる全node無効化を明記しました。CPUでも旧設定値の一致とdigestの変化を確認しました。

コード変更・commit・pushはありません。branch `codex/court-video-shared`、HEAD `a1b7029c5`、Ready for reviewを維持しています。merge・closeは行っていません。