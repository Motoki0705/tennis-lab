③の本番再開と起動受入を完了しました。学習はsystemd配下で継続中です。

- branch: `codex/ball-query-desktop`
- commit: `f06f1bda6610838313e4b306a92a4db5e42ed579`
- draft PR: [#1931](https://github.com/Motoki0705/tennis-lab/pull/1931)

前半6,000更新の固定validationは128,668 frames／位置教師85,473で旧基準とIDが一致しました。225.46 MBの周期snapshotをDriveから独立復元し、model・AdamW・RNG・schedule・sample・logの一致を確認しました。後半628更新まで受け入れ、600→601のCNN解除、LR倍率0.1、有限loss／勾配も確認済みです。

本番jobは `0-8205507466335230348_2026-11-12T14-08-53-664769651Z_1594191_ball-query-length-desktop-production-a1`、resource all。service PID 1557549／PPID 278、trainer PID 1594405／PPID・PGID 1594280です。15:13 UTCの確認で生存し、異常イベントはありません。

実測は解除後0.723秒[local path]、学習stepのpeak VRAM allocated 10.43／reserved 11.17 GiB、trainer peak RSS 4.24 GiB、host available最小12.44 GiB。受入時点の残り5,372更新は、演算のみ約65分、保存・validationを加えた概算約100分です。

CPU17件・実artifact受入検査・Ruff・knowledge検証が成功。最新HEADのPython CI／Knowledge CIも成功しました。pre-commitは実施済みでhook対象なし。validatorは指定・試行・完了とも0回です。

[受入サマリ](https://github.com/Motoki0705/tennis-lab/blob/f06f1bda6610838313e4b306a92a4db5e42ed579/knowledge/runs/run-ball-query-desktop-20261112/startup-acceptance-summary.json)・[実測と手順](https://github.com/Motoki0705/tennis-lab/blob/f06f1bda6610838313e4b306a92a4db5e42ed579/knowledge/runs/run-ball-query-desktop-20261112/README.md)・[復元手順](https://github.com/Motoki0705/tennis-lab/blob/f06f1bda6610838313e4b306a92a4db5e42ed579/knowledge/runs/run-ball-query-desktop-20261112/recovery.md)を保存し、adapter・command・環境・テスト・受入記録のDrive保存と独立readbackも完了しています。

定期的な学習進捗確認は終了しました。terminal eventを保持しています。残作業は後半完了、query累積12kの終端評価・保存、4条件比較とknowledge結果記録です。稼働中のworktree・source・dataset・envは保持してください。merge／Issue closeは行っていません。