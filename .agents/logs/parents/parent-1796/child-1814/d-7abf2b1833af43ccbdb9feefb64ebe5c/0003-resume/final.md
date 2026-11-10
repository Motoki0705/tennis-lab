[PR #1865](https://github.com/Motoki0705/tennis-lab/pull/1865)を更新しました。branchは`codex/decision-log-contract`、HEADは`0c91d16d6`です。

- **1**: envelopeと変換条件を固定し、再renderによるcapture消失を解消。
- **2**: 祖先の起動事実を保持し、unknown経由の逆戻りを拒否。6ケース成功。
- **3**: 全incoming参照を調べ、行動・比較分岐が参照する途中結果を保護。3ケース成功。
- **4**: `worker_logs/safety.py`の変更を撤回。旧ready/pendingログの再開2ケース成功。

回帰13件は修正前8失敗・5成功から全成功へ改善。対象全279件、pre-commit、ruff/mypy、[Python CI](https://github.com/Motoki0705/tennis-lab/actions/runs/42809567493/job/129819996877)が成功しました。CIはAPT準備失敗分のみ再試行しています。

worktreeはcleanです。親による解消確認・merge・merge後main確認は未実施です。