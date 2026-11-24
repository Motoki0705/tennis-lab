CI失敗への修正を同じブランチにpushし、実行証拠を更新しました。 [PR #2448](https://github.com/Motoki0705/tennis-lab/pull/2448)

- **FP32互換性**。導入時のソース指紋をfixtureに固定し、後続の台帳編集を旧キャッシュへ正規化しないことも検証しました。
- **掲示板ブラウザテスト**。旧版のローカル単発実行は通過し、並行試験で模擬Herdrの共有JSON破損を再現しました。fixtureの読み書きを直列化した後は、4並列・計800要求が成功し、呼び出し記録も800件揃いました。

実Chromiumを含む関連テスト **279件が282.58秒で成功**し、pre-commitも通過しました。ブランチは `codex/parent-1739-issue-2442-primary`、新HEADは `7c1ad6d989f0ed6c6bd9ef81e6347753fa43349f` です。GitHub CIの結果を親CLIへ引き継ぎます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2448,"head":"7c1ad6d989f0ed6c6bd9ef81e6347753fa43349f","resume_on_success":false}}