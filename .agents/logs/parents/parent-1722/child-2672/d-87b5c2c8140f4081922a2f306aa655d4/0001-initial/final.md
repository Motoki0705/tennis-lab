`inventory.py` の SHA-256 に合わせて、multiview 互換表の `after` 1項目を更新しました。`post_train` の追加登録は CLI のパス検証用で、シーン計算・公開処理に影響しないことを差分で確認しました。[PR #2673](https://github.com/Motoki0705/tennis-lab/pull/2673)

- 最終 HEAD で指定の統合テスト15件が通過しました（23.13秒）。pre-commit は終了値0で、Ruff と mypy は JSON 差分のため対象外でした。
- 現行 HEAD の multiview ソース識別は全16設定で基準版と一致しました。`event_detection` の互換表は #2462 由来の公開処理のハッシュ差で適用条件を外れます。そのハッシュを記録版にそろえた検証では、両表が順に適用され、全16設定でイベント側の基準識別と一致しました。[#2462](https://github.com/Motoki0705/tennis-lab/pull/2462)

ブランチは `codex/parent-1722-issue-2672-primary`、コミットは `c11ffa6e6db5a4fe3c991eb111ade6e73f7f034a` です。PR CI は実行中で、main CI はマージ後の確認待ちです。[CI run](https://github.com/Motoki0705/tennis-lab/actions/runs/45887610669)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2673,"head":"c11ffa6e6db5a4fe3c991eb111ade6e73f7f034a","resume_on_success":false}}