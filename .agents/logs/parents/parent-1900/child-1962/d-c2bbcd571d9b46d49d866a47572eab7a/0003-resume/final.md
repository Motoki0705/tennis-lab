cross-1を修正し、[PR #1973](https://github.com/Motoki0705/tennis-lab/pull/1973)へpushしました。

- branch: `codex/log-turn-traceability`
- 最終HEAD: `895c341926f71cffbabe138ca51cd0368e41c6c8`
- [CI #4094](https://github.com/Motoki0705/tennis-lab/actions/runs/43563468891)：全10 CPU・Lint・system dependency・集約Python成功
- 関連テスト：**441 passed（138.74秒、`-n 2`）**。pre-commit、変更opsのmypy／ruff成功。worktreeはcleanです。

termination観測version 2で#1984のcheckpointを直接受け取り、既知残件と未確認値を併存させます。旧version 1観測・公開原文は維持しています。

| #1984の契約 | 対応 |
|---|---|
| RemainingPublication全12フィールド | 同じキー・値と`None`を保持 |
| public_key | workerは`d-`付き、decisionはbatch_keyまたは`None` |
| unpublished_count=None | 「未公開記録数: 未確認」と表示 |
| remaining_complete=false | 既知残件を残し、一覧の照合を「未確認」と表示。空一覧もゼロと認定しない |
| error | checkpoint・残件それぞれのエラーを保持・表示 |
| publication_actor | 整数の親番号／公開文字列を受け入れる |
| resume_command／decision_resume_command | 両再開先を保持・表示し、安全検査を適用 |

追加・拡張したテストは次のとおりです。増分は計10ケースです。

- `test_termination_preserves_unknown_counts_and_known_publications`：3ケース
- `test_termination_inventory_rejects_invalid_producer_fields`：6ケース
- `test_typed_termination_cannot_publish_a_private_resume_path`：decision再開先の1ケース追加

修正前の3件失敗を再現し、修正後は成功しました。#1984実関数の出力を本adapterへ渡す互換性確認も成功しています。実producerでは件数`None`でも`remaining_complete=true`になるため、両者を独立に扱っています。

release／handoff／supervisorへの配線は指示どおり未実施です。#1984 merge後の接続と、merge後mainの表示確認が残っています。