[PR #2208](https://github.com/Motoki0705/tennis-lab/pull/2208) を作成しました。branch: `codex/storage-tiering`、HEAD: `e9874f353`。本文は `Refs #1990` を維持しています。

- 自己exec実験で「fdはEACCES、mapsは可読」を1回再現。カーネル内部の原因は未特定です。
- PID/starttimeを固定し、50ms間隔で最大3回再確認します。全参照を再検査し、未証明の生存個体はunknownを維持。履歴も監査出力へ残します。
- 修正前の9ケースと追加の個体照合2ケースの失敗を確認後、修正しました。新規19ケース、関連 **6,597件成功**（39分18秒）。pre-commit成功。
- [対象HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/43906242173) はPythonを含む全14ジョブ成功。最長CPU shardは **27分7秒**。
- 実機scan・引数なしcycle dry-run各10回で、最終版はunknown・failedとも0件。31個体の終了を再確認できました。修正前はcycle 4/10回、6単位が失敗していました。

[実行記録](https://github.com/Motoki0705/tennis-lab/blob/e9874f353ddf2ebf6a44e494472d3d5e54b1d454/tmp/storage-proc-recheck/report.md)。実機の配置・timer・sudoは未変更です。親側のレビュー・merge・main反映が残ります。