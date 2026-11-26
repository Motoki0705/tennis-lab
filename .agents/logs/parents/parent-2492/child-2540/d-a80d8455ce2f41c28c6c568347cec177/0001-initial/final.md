論文フィルタの期待件数と行 ID を、ノードの YAML から求めるようにしました。再読み込み後も、選択した WASB の詳細と、GVHMR で絞り込んだ一覧がそれぞれ復元されることを検証します。

手元の Chromium で修正前の「期待 3 件、実際 11 件」を再現しました。修正後は実データ 739 件中の該当 11 件で対象テストが通り、ブラウザ全 11 件、単体テスト 210 件、lint、ビルドも通過しました。ブラウザ全体は既存のシステムライブラリを使って直接実行し、84.9 秒でした。pre-commit は終了コード 0 で、ruff と mypy は対象ファイルなしでスキップされました。

`codex/parent-2492-issue-2540-primary` に `76df5a78c42f701fc66357f9a852620c9179ee2f` を push しました。PR の CI は実行中です。PR とマージ後の main の `knowledge-webui` の結果が確認待ちです。[PR #2541](https://github.com/Motoki0705/tennis-lab/pull/2541)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2541,"head":"76df5a78c42f701fc66357f9a852620c9179ee2f","resume_on_success":false}}