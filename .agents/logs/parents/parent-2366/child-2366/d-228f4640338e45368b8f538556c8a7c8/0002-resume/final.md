手動workerの起動テストに一時ミラーを用意し、E:への出力配置を検証できるようにしました。PR #2371 の既存修正を取り込み、同じブランチ `codex/parent-2366-issue-2366-primary` にpushしました。[PR #2376](https://github.com/Motoki0705/tennis-lab/pull/2376)

CIで失敗したケースをローカルで再現し、修正後は関連64件がすべて通過しました。Ruff、mypy、pre-commitも通過しています。新HEAD `f32bf0211` のPython CI結果待ちです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2376,"head":"f32bf0211428155e110ba98279cef5ae1260aea2","resume_on_success":false}}