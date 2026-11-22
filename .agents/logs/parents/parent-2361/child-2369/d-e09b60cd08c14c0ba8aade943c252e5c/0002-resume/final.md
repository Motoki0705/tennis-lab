レビュー指摘2件を修正してpushし、マージ後の手順を更新しました。[PR #2423](https://github.com/Motoki0705/tennis-lab/pull/2423)

- `spin ops status` と `on` は、読み込み済みserviceの `OnFailure` と通知インスタンスの `LoadState=loaded` を確認します。照合に失敗すると理由を返し、`on` は有効化前に停止します。
- 通知unitは、未送信確定と、送信結果不明で再送時に重複する可能性がある場合をjournalで区別します。どちらも終了値1です。

修正前に再現した6ケースは修正後にすべて通過し、関連pytest 484件とpre-commitも通過しました。実機のtimerは `linked` / `inactive` です。本番cycle、auditの照合、timerの有効化はマージ後に残っています。

ブランチは `codex/parent-2361-issue-2369-primary`、HEADは `6452faaf0148dfe12dfd658c647f2e7397324356` です。CIは親の監視に引き継ぎます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"ci","pr":2423,"head":"6452faaf0148dfe12dfd658c647f2e7397324356","resume_on_success":false}}