# ワーカーの委任と統合

担当Issueとワーカーは別々に決める。例えば「推論高速化と品質維持」は一つの成果として、実装と品質比較を分担できる。比較で劣化が出たら同じIssueに原因調査を追加する。比較結果によって採用手法を決めてから本番導入する場合は、採否を分割の境界にできる。

| 場面 | 操作 |
|---|---|
| 担当を用意 | 親だけなら自身の番号を `--child` に使う。最初の委任が `primary`。各担当に同じ基準commitを含む専用branchとworktreeを作る。依存PRがあれば `launch_plan.stacked_on` のheadを含める。 |
| 手動親から起動 | `delegate` で予約し、`run-worker --prompt-file` で実行。補助担当は両方に `--worker 名前` を付ける。独立したrunは背景で並行させる。 |
| 自律親から起動 | `start-worker` で予約し、supervisorに起動を渡す。補助担当も1枠使う。pytestは共有lockなしで並行実行できる。GPU・Blenderはtraining queueの枠を使う。 |
| 結果を受ける | `wait-worker` または `inspect-worker` で担当と終了・回収を照合し、`final_file` と成果物を読む。複数人のときは統合担当にも `--worker primary` を指定する。 |
| 調査を追加・継続 | 調査専用は `--planning`。同じ担当への続きは手動の `run-worker --resume`、自律の `continue-worker`。計画に回答を結び付ける場合は `decide-worker-plan`。 |
| 統合 | 補助担当は結論と根拠、またはcommitを返す。統合担当が競合を解消し、統合HEADを検証してIssueを閉じるPRを持つ。親は `record-result` に受入の証拠を残す。 |

ワーカーはSol固定。PR作成後も補助担当を追加できる。個別停止は `stop-worker --child N --worker 名前`、Issueそのものの保留・再開は `stop` / `resume` を使う。
