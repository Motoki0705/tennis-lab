---
name: parent-session
description: 親issueの担当・実装・調査・引継ぎを進める。親AIが合意と状況から計画を判断し、CLIで担当・worker・証拠・進捗を管理する。説明・参照・起票だけの依頼には使わない。
---

# 親issueを進める

## 担当と実行根拠

`status` で担当・既存PR・委任を確認する。新規は `register --summary`、交代は `handoff --accept`。目的・範囲・期待動作・受入条件と人間の元発言を `agree` に残す。dispatcherから起動した親は予約を受理し、[自律workflow](../../ops/parent/development_workflow.md) の実行根拠を使う。既に合意した判断は聞き直さない。
担当表示・親間連絡には [表示](../herdr-session-label/SKILL.md)・[掲示板](../parent-board/SKILL.md) を使う。

## 親が判断すること

一つの成果と受入条件を一つのIssueに置く。「この結果で次の目標・着手が決まる」という依存境界で子を作る。同じ目標の調査・実装・比較テストは同じIssueで並行できる。予想外の結果が出たら、そのIssueに追加検証を委任する。未分割なら親自身を担当Issueにできる。

### 合意と計画の変更

目的・範囲・受入条件・権限の実質変更は人間と確認する。順序・担当・調査方法の変更は合意の範囲で親が判断する。変更の影響を受けた証拠だけを確かめ直す。次の操作を決めた理由は既存の結果を参照して [判断記録](../../ops/decision_logs/README.ja.md) の `decide` に残す。

## 委任と継続

親は直接実装するか、統合担当に成果PRを任せる。並列に進める利点があれば、担当範囲・共通の基準commit・返す成果を指定し、各ワーカーに専用branchとworktreeを用意する。統合担当は補助ワーカーのcommitや測定結果を取り込み、競合を解消して統合HEADを検証する。

手動親の例。以降の操作では `--child` が担当Issue、`--worker` がその中の担当名を表す。

```bash
parent=(.venv/bin/python -m parent --parent 1216)
export PYTHONPATH=.agents/ops
"${parent[@]}" delegate --child 1217 --agent codex --branch codex/impl --worktree "$impl_tree"
"${parent[@]}" delegate --child 1217 --worker quality --agent codex --branch codex/quality --worktree "$quality_tree"
"${parent[@]}" run-worker --child 1217 --worker quality --prompt-file tmp/compare.md
"${parent[@]}" run-worker --child 1217 --worker primary --prompt-file tmp/integrate.md
"${parent[@]}" review-request --prs 1250 --instruction-file tmp/review.md --evidence-file tmp/tests.md --wait
```

独立した実行は背景で並行し、`wait-worker` で終了と回収を待って `final_file` を読む。自律親では `start-worker` と `continue-worker` を使う。調査専用ターン、停止、資源枠、再開・引継ぎは [親CLI](../../ops/parent/README.md#手動workerの起動継続とrun記録) を参照する。

## 成果とレビュー

原要件と受入条件を渡し、PRにつき一度だけ正式レビューを依頼する。majorがあれば、統合担当が各指摘に修正commitとテストを対応付ける。親は元の指摘と証拠を照合し、解消または対象外の理由を記録して `merge-continue` する。現HEADのCI・依存・権限などが成立すればCLIがマージと事後確認を進める。[停止と復旧](../../ops/review/README.md)

## 親Issueへ伝えること

人間の判断依頼、実質的な合意変更、親全体の進め方を変える発見、最終成果を `notify` で伝える。定型の経過はCLIの証拠記録・子・PRに集める。全体の受入条件、マージ後のCI、ワーカーの終了、ログの公開を照合する。今回の受入結果と次回改善を分けて [構造化finish](../../ops/parent/improvements.md) に渡す。
