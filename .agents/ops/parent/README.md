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

証拠用Issueの作成直後は、POST応答のIssue番号をjournalに保存し、番号を指定して取得した結果と照合する。
ラベル付き一覧への反映前でも、直接照合が成功し、一覧に競合する紐付けがなければ確定できる。

状態ルートは実ディレクトリを指定する。状態ルートと `parents/` は実行ユーザーが所有し、上位は root または実行ユーザーが所有する。
状態ディレクトリ本体と状態ファイルにはグループ・その他への書込み権限を与えない。
上位ディレクトリのグループ書込みは、その他への書込み権限がなく、実行ユーザーが所有し、
ディレクトリのグループが本人だけの主グループであることを主・補助メンバーの照合で確認できた場合に許可する。
`/tmp` のようなその他も書込み可能な上位ディレクトリは、sticky bit がある場合に許可し、次のパス要素の所有者を照合する。
registry lock と読み書きは同じディレクトリに固定し、名前が別の実体を指したら処理を中止する。
v2 の更新は索引と親レコードの新しい内容を先に準備する。置換に失敗した場合は、同じロックの内側で置換済みの親レコードを元に戻す。

## 実装の入口

- [session.py](session.py) と [execution.py](execution.py) が担当と実行根拠を照合する。
- [launch.py](launch.py)・[manual_workers.py](manual_workers.py)・[development.py](development.py) が委任予約、手動実行、dispatcherとの接続を扱う。
- [local_jobs.py](local_jobs.py) は、GPU不要の長時間ローカルコマンドを独立したsystemd user transient service（`tennis-lab-local-job-<job_id>`）として実行し、`PARENT_LOCAL_JOBS_DIR` または `<state_dir>/local-jobs/<job_id>/` に実行記録・ログ・完了記録を保存する。
- [worker_monitor.py](worker_monitor.py) は、固定した起動元HEADから運用コードを一時のdetached worktreeに展開するsystemd user serviceを起動する。service停止時にworktreeを回収する。監視中のレジストリ処理はGit共通lockを使い、CI・queue・`local` jobや空き枠を待つ間はlockを解放する。queueの終端ファイルから60秒以内、`local` jobのプロセス消滅から10秒以内に完了記録を観測できなければ、`wait-worker` にエラーを返す。
- [review_handoff.py](review_handoff.py) が [マージ自動進行](../review/README.md) へ渡す。
- [completion.py](completion.py) と [improvements.py](improvements.py) が全体の受入・次回改善を記録する。改善提案の期限処理は [systemd](systemd/) にある。

`run-worker` が `capacity_deferred` と終了コード75を返した場合は、容量が戻ってから同じ委任で
初回コマンドを再実行するか、最後に実行したターンから `--resume` する。CLI未起動の証拠は
`refused_run_dir` に残る。自動再開の容量拒否では `followup` の `observation` に観測結果を残し、
親へ再開を引き渡す。

## agyの手動親

手動起動したAntigravity CLIの対話ルートで `register`、`status`、`agree`、`run-worker`、`inspect-worker`、`wait-worker`、`handoff`、`release` などの手動親操作を使える。
ホスト側の `run_command` から実行すると、[共通identity](../lib/identity.py)が環境の会話IDを
単一の前景TUI祖先とその排他ロックに照合する。[agy_session.py](../lib/agy_session.py)はさらに
そのTUIのローカル読み取りRPCでルート会話IDと実行trajectoryを照合する。
子会話を別paneで開き直した場合も、ルートの確認に通らなければ担当操作を拒否する。
print、ACP、入れ子、子agent、workflowからの担当操作、ホスト祖先を確認できないterminal sandboxや `setsid ... &` で切り離したシェルは対象外。
`run-worker` / `wait-worker` の直接非同期起動と `inspect-worker` 確認手順は[ワーカーの委任と統合（agy手動親での起動と監視）](../../skills/parent-session/references/workers.md#agy手動親での起動と監視)を参照。
担当登録・引継ぎ受領に使う起動と再開の手順は[手動親ランチャー](../herdr_labels/README.md#agyの手動起動と表示)を参照。
agyの識別情報がある `status` は、ホスト祖先・会話ロック・trajectoryの照合異常をエラーとして返す。
自律親の起動範囲は[dispatcher](../dispatcher/README.md#担当実行枠中断復旧)を参照。

## 改善提案の期限

期限処理は日次jobとしてcheckoutを固定する。手元でhelpを読む場合も、repoのcheckoutで同じ入口を使う。

```bash
PYTHONPATH=.agents/ops .venv/bin/python .agents/ops/checkout_job.py --repo . -- .venv/bin/python -m parent improvement-expire --help
```

末尾を `--dry-run` にすると期限対象を確認できる。作成から7日経過した未取り出しのopen提案を `not_planned` で閉じる。日次unitの導入・有効化は [ops control](../ops_control/README.md) の `proposal-expiry` を使う。
