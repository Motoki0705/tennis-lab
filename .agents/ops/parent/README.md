# 親セッションCLI

親AIの手順と状況別の参照先は [parent-session](../../skills/parent-session/SKILL.md)。引数と入力例は `PYTHONPATH=.agents/ops .venv/bin/python -m parent --help` および各サブコマンドの `--help` で確認する。

## 責務と状態の正本

| 保存先 | 管理する事実 |
|---|---|
| GitHub Issue・PR | 要件、正式な親子関係、受入条件、closing対象、マージ結果。書式は [issues](../issues/README.md) を参照。 |
| ローカル `parent-sessions/` | `registry.json` と `parents/` が担当・世代・委任を保持する。runと自動進行には固有のIDを使う。[state.py](state.py) |
| 証拠用Issue | 機械イベントと人間の元発言への参照。保存先のbindingと公開再送を照合する。[evidence.py](evidence.py) |
| ワーカー・親のログ | [ワーカーの各ターン](../worker_logs/README.md) と [親の判断](../decision_logs/README.ja.md) を別々に保存する。 |

## 排他とI/O

担当変更と掲示板配送は同じownership fenceを使う。取得順は操作lock、ownership、registry。GitHub・runnerへのI/O中はregistryを解放し、復帰後に担当・対象記録を照合する。読取はsnapshotを使う。版移行と競合時の再検証は [state.py](state.py)、GitHubの取得範囲とキャッシュ境界は [github.py](github.py) を参照。

## 実装の入口

- [session.py](session.py) と [execution.py](execution.py) が担当と実行根拠を照合する。
- [launch.py](launch.py)・[manual_workers.py](manual_workers.py)・[development.py](development.py) が委任予約、手動実行、dispatcherとの接続を扱う。
- [review_handoff.py](review_handoff.py) が [マージ自動進行](../review/README.md) へ渡す。
- [completion.py](completion.py) と [improvements.py](improvements.py) が全体の受入・次回改善を記録する。改善提案の期限処理は [systemd](systemd/) にある。

## 改善提案の期限

期限処理は日次jobとしてcheckoutを固定する。手元でhelpを読む場合も、repoのcheckoutで同じ入口を使う。

```bash
PYTHONPATH=.agents/ops .venv/bin/python .agents/ops/checkout_job.py --repo . -- .venv/bin/python -m parent improvement-expire --help
```

末尾を `--dry-run` にすると期限対象を確認できる。作成から7日経過した未取り出しのopen提案を `not_planned` で閉じる。日次unitの導入・有効化は [ops control](../ops_control/README.md) の `proposal-expiry` を使う。
