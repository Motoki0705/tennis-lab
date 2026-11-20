PR #2346 のレビュー指摘6件を、同じブランチ `codex/parent-2312-issue-2326-primary` のコミット `bdc04ead62f020b0fb7ecd3f742228dcfba70cc7` で修正しました。[PRの検証コメント](https://github.com/Motoki0705/tennis-lab/pull/2346#issuecomment-7048403542)

1. 公開を観測時の `main` SHA に固定し、残す側のエントリが並行削除された場合に公開を拒否しました。
2. Codex の読取範囲を制限し、実機CLIと模擬応答で元checkoutの専用memoryと未コミットファイルの読取拒否を確認しました。
3. 公開インテントを永続化し、応答喪失・SIGTERM・片付け失敗からの復旧を確認しました。未反映の候補は照合待ちとして保持します。
4. runner と service に有限のタイムアウトを設定し、手動・service 両経路でロック、子孫プロセス、一時展開の回収を確認しました。
5. `citations` の欠落だけで即時発火することを確認しました。
6. 破損状態の原本を保全し、`spin ops status` に失敗理由と退避先が出ることを確認しました。

追加16件を含む pytest **620件**、共有memory検査、Ruff、mypy、pre-commit、systemd 構文検査が通過しました。新HEADのCIは起動済みで、`Python` の結果確認を親へ引き継ぎます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2346,"head":"bdc04ead62f020b0fb7ecd3f742228dcfba70cc7","resume_on_success":false}}