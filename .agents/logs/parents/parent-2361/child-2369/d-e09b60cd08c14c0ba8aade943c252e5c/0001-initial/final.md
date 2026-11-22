#2369 の実装をpushしました。[PR #2423](https://github.com/Motoki0705/tennis-lab/pull/2423)

ブランチは `codex/parent-2361-issue-2369-primary`、HEADは `a1d19c17e30673236c2d8539b37fe424c1db537c` です。

- 失敗通知を他のops unitでも使えるテンプレートにしました。実機の一時user unitで失敗を注入し、Windowsトーストの送信1回と、送信失敗時に対象unit名と原因がjournalに残ることを確認しました。
- 容量gateが終了値75を返す設定で、実機の `cycle --dry-run` は28単位を検査し、終了値0でした。
- 関連pytest 401件、pre-commit、systemdのunit検証が通過しました。クラッシュ・復旧境界の全組み合わせはローカルで未完走です。

本番timerは `linked` / `inactive` です。マージ後に親が本番cycleとauditを確認し、timerを有効化する手順をPRに記載しました。CIは親の監視に引き継ぎます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2423,"head":"a1d19c17e30673236c2d8539b37fe424c1db537c","resume_on_success":false}}