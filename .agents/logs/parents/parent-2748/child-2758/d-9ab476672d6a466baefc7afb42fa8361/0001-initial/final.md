[PR #2767](https://github.com/Motoki0705/tennis-lab/pull/2767) を ready で作成しました。`Closes #2758`、検証コマンド、入力・出力の SHA-256 を PR 本文に記録しました。

ブランチ `codex/parent-2748-issue-2758-primary`  
HEAD `521bc7a76da04296e7a0a0ef971313a76459ddd6`

- 全69クリップの開始・終了時刻は最大誤差0秒でした。train / val / test は47 / 11 / 11、スコア連鎖68接続と試合終了を確認しました。
- スコア前後・サーバー・勝者は各69/69件。終了原因は `net_error` 17/69件で、52件は `unknown` です。実データに手動補足はなく、UE/FE と手動サーブ試行の利用可能率は各0/69件でした。
- S1 が読み込む829ファイルの SHA-256 を照合しました。PostgreSQL 17.12 に採用620件・除外293件を投入し、`clip_009/e003` が採用済みかつポイント外として保存されることを確認しました。

[JSON 監査]([local path])、[CSV 監査]([local path])、再投入後の DB エクスポートは、それぞれ2回の出力がバイト単位で一致しました。単体テスト190件、PostgreSQL 結合テスト10件、pre-commit の Ruff・MyPy が通過し、専用 Compose は停止済みです。[PR CI](https://github.com/Motoki0705/tennis-lab/actions/runs/46175974313) は実行中です。

PARENT_RESULT {"schema_version":1,"state":"done","wait":null}