指摘`pr-1942-1`を修正し、[PR #1942](https://github.com/Motoki0705/tennis-lab/pull/1942)へpushしました。

- branch: `codex/issue-standalone-label`
- HEAD: `f9ca0826841ebd9d3f9b8a8222120c57c066a5fc`
- 親参照・子一覧を行走査で検出し、重複節でも根拠を保持します。standaloneの節解析失敗と行走査失敗は`body`問題として報告します。
- 親子の根拠がない無ラベルIssueの重複節は従来どおり許容します。行走査不能な未閉鎖ブロックは、無ラベルでも`body`問題とする理由を規約・PRに記録しました。
- 修正前は追加9ケース中8件失敗。修正後は重点108件、関連636件すべて成功（428.59秒）。pre-commitと変更モジュール単体のmypyも成功しました。
- 対象HEADの[CI `Python`は成功](https://github.com/Motoki0705/tennis-lab/actions/runs/43352772636/job/131653926879)。CPU全10分割・Lintも成功しています。

PR本文の証拠を更新済みで、worktreeはcleanです。指摘解消登録・merge・merge後の確認は親に残しています。