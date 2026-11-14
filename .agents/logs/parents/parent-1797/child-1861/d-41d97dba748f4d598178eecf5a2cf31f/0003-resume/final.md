[PR #1872](https://github.com/Motoki0705/tennis-lab/pull/1872)へpush済みです。branch: `codex/review-reuse-adoption`、commit: `167876472`。

- `close-interrupted`を追加。lock取得・runtime cleanup・workspace撤去を確認し、未公開の終結状態にします。承認・coverageには使わず、後続requestの履歴を保持します。照合表示にも対応しました。
- 追加10 node・13ケース。関連52ケース成功、pre-commit成功。cleanup検査を外す故障注入も検出しました。
- [新HEADのCI](https://github.com/Motoki0705/tennis-lab/actions/runs/42841807419/attempts/2)はLint・CPU全10レーン・Python成功。初回apt失敗は、ログ確認後の再実行1回で解消しました。

README・PR本文・子issueを更新済みです。残作業は親の正式レビュー・merge判断・merge後main確認です。