CI の `KeyError: 'runs'` を修正しました。引継ぎテストで `enabled` だけを返していたモックを、通常の初期化で保存した無効状態のスケジュールに置き換えました。

失敗ケースを含む関連テスト 142 件と pre-commit の Ruff・mypy が通過しました。コミット `57c634228` を同じブランチに push し、検証結果を追記しました。新しい HEAD の CI 結果は未確認です。[PR #2399](https://github.com/Motoki0705/tennis-lab/pull/2399)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2399,"head":"57c6342283d7d0e2f7c70d6c66d778e2628f0327","resume_on_success":false}}