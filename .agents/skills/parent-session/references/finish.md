# 記録と終了

| 伝える相手・目的 | 書き先 |
|---|---|
| 子の局所的な観測・検証 | 子IssueまたはPR。機械証拠はCLIが保存し、自由文の根拠は `note` に残す。 |
| 観測を受けて親が次の手を決めた | `decide` に元の結果と判断を結び付ける。[入力形式](../../../ops/decision_logs/README.ja.md#親の判断を明示記録する) |
| 人間への質問、実質的な合意変更、全体を変える発見、最終成果 | `notify`。同じ意味の通知は同じkey・specを使い、完了通知のkeyは `completion`。最終報告をここに集約する。 |

### 全体の受入

`status.completion.requirements` の `basis_ref` と全条件のIDを使い、各項目の達成・根拠、残る必須作業、改善候補をfinishに記入する。completedには全項目の根拠と空の `remaining_required` が要る。手動親は `release --ended --spec`、自律親は `workflow-report --spec`。JSON例は各コマンドのhelpを参照。

### 次回改善

今回の未達は受入結果に残す。独立した改善だけを `improvement --key` で記録し、`improvement-propose` で元親ごとの提案にまとめる。人間の判断を `improvement-decide` に保存し、取り出しを指示されたら `improvement-extract` で承認分を一つの親へ移す。元親の完了はこの判断を待つ必要がない。
