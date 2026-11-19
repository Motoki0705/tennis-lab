# 記録と終了

| 何が変わったか | 記録先 |
|---|---|
| 子の局所的な観測・検証 | 子IssueまたはPR。自由文の根拠は `note` に残せる。機械証拠はCLIが親ごとの証拠用Issueに保存する。 |
| 結果を受けて親が次の行動を選んだ | `decide` に観測結果と判断を結び、返された判断IDを対応する操作へ渡す。保存済みnoteや計画回答は参照で使える。[判断の入力形式](../../../ops/decision_logs/README.ja.md#親の判断を明示記録する) |
| 人間への質問、実質的な合意変更、全体を変える発見、最終成果 | `notify`。質問には回答待ちの範囲、方針変更には影響先と次の操作を書く。`evidence` はこの親の検証済み機械記録のコメントURL。同じ内容の再送は同じkey・specを使う。 |
| Issue・Projectの現在地 | `sync` はGitHubの実状態から親本文を更新する。Projectを運用している場合は節目を `project-sync` でプレビューし、内容を確認して同じspecを `--apply` する。[Projectの入力と節目](../../../ops/project_progress/README.md#導入更新の入口) |

## 全体の受入

1. マージ済みPRと対応するIssue、マージ後確認、未終了ワーカーを照合する。取消・duplicateの子は理由と引継ぎ先を確認し、達成した成果から区別する。ログの公開は `saved_to_main` 等で別途確認し、未公開分は再開先を残す。
2. `status.completion.requirements` の `available`、`basis_ref`、`criteria` を読む。**全IDを一度ずつ**finishの `acceptance` に入れ、項目ごとに根拠を付ける。`available: false` のときは `reason` を読み、本文の受入条件と合意の差を解消してから再取得する。未確認は `unverified`、未達は `unmet` として残す。
3. 全項目が `met`、`remaining_required` が空になったら、`notify` の固定key `completion` で成果・残件・成果物URLを報告する。`status.notifications` で `published` を確認する。`pending` は証拠側への保存までなので同じspecで通知を完了する。
4. 手動親は `release --ended --spec finish.json`。自律親は `workflow-report --spec report.json` のversion 2入力に同じfinishを入れる。JSON例はそれぞれのhelpにある。担当を渡す段階なら、残件と再開先を [引継ぎ](ownership.md#引継ぎ) に残す。

## 次回改善

今回の未達は `remaining_required` に残す。独立した改善候補は `improvement --key` に根拠と範囲を記録し、finishの `improvements` で全候補を参照する。候補なしも `reviewed: true, candidates: []` を明示する。

候補をまとめて `improvement-propose`。人間の判断は元発言・出典・対象revisionとともに `improvement-decide` に保存する。人間が取り出しを指示したら `improvement-extract` で承認分を一つの親へ移し、新しい親は `agree --improvement-approval` で根拠を受ける。元親は次回改善の判断待ちを理由に再開しない。
