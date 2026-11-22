{
  "schema_version": 1,
  "repair_id": "[private identifier]",
  "pr": 2397,
  "outcome": "repaired",
  "head_sha": "0893502a670e6f4877aba17b8e4cfa0c730919b5",
  "summary": "main との競合を解消し、同じ PR ブランチに push しました。容量不足からの同一ラウンド再開と、実行障害後の次ラウンドへの明示再試行を統合し、試行ごとの終了理由を確認できるようにしました。",
  "verification": [
    "関連 pytest 372 件が通過しました。容量不足から同一ラウンドで再開し、実行障害後に明示再試行する回帰テストを含みます。",
    "ruff、mypy、pre-commit、マージ元との差分の git diff --check が通過しました。",
    "git merge-tree で取得済みの origin/main（8bd5c897b）との競合が解消されたことを確認しました。",
    "新 HEAD の GitHub Actions CI は呼出側での確認待ちです。"
  ],
  "parent_request": null
}