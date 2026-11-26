#2427 を実装し、`Closes #2427` を含む PR を作成しました。 [PR #2481](https://github.com/Motoki0705/tennis-lab/pull/2481)

- `wait-worker`、補足待ち、マージ進行の外部待機中は共有 lock を解放します。復帰時に lock を取り直し、完全 HEAD が変わっていれば旧・新 HEAD を表示して75で終了します。
- git-sync は3回連続の見送りで、回数・経過時間・最後に観測した lock 保持者を journal と Windows 通知に出します。
- 一時 checkout で、queue 待機中に git-sync が `unchanged` で完了し、その後 `wait-worker` が job の完了を受け取ることを確認しました。別ケースでは fast-forward 後の終了コード75を確認しました。

関連テスト460件が通過し、最終調整後にマージ進行の26件も再検証しました。pre-commit の ruff と mypy は通過済みです。Windows 通知はモックで検証し、マージ後に親が行う実機 journal の確認手順を PR に記載しました。

ブランチは `codex/parent-2474-issue-2427-primary`、完全 HEAD は `fd1d8e6bba067518e689fede7092b7466af49825` です。CI の起動を確認し、結果の監視を親CLIへ引き渡します。 [CI 実行](https://github.com/Motoki0705/tennis-lab/actions/runs/45348011396)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2481,"head":"fd1d8e6bba067518e689fede7092b7466af49825","resume_on_success":false}}