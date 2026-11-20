# ワーカーの委任と統合

## Issueと担当を決める

「推論を高速化し品質を維持する」は一つの成果。実装と品質比較を別ワーカーに任せ、劣化が見つかれば同じIssueに調査担当を足せる。一方、比較で手法の採否を決めてから本番導入するなら、採否を確定する地点で子Issueを分ける。

正式な直属子があれば `--child` は対象の子番号。上位親も直属子もないIssueでは、自身の番号を指定する。親が直接実装する場合は専用worktreeで変更し、作成したPRに受入条件ごとの証拠を揃える。新しい子を作る場合は、受入条件と先行依存を [子Issueの書式](../../../ops/issues/README.md#本文) で定める。同じ子の作成を再送するときは同じkeyとspecを使う。

## 同じIssueに二人を割り当てる例

親1216・担当Issue1217の例。両担当の基準commitを固定する。`--from` が担当別のbranch・worktree・共有入力・venv・出力を準備し、予約と初回起動を行う。メインcheckoutの `.claude/worktrees/` に作成する。各担当への指示に作業範囲・受入条件・返すcommitまたは測定結果を書く。

```bash
export PYTHONPATH=.agents/ops
p=(.venv/bin/python -m parent --parent 1216)
base=$(git rev-parse origin/main)
run_logs=$(mktemp -d)
setsid "${p[@]}" run-worker --child 1217 --worker primary --from "$base" --prompt-file impl.md </dev/null >"$run_logs/impl.log" 2>&1 &
```

primaryの予約を `status` で確認したら、実装の終了を待たずに補助担当を起動する。

```bash
setsid "${p[@]}" run-worker --child 1217 --worker quality --from "$base" --prompt-file compare.md </dev/null >"$run_logs/quality.log" 2>&1 &
```

自律親は各担当に `start-worker --from 基準commit` を使う。ワーカーはSol、通常max。補助担当も1枠を使い、GPU・Blenderの使用はtraining queueで調停する。

## 待つ、追加する、受け入れる

- `launch_plan` の `action=ready`、既存予約、依存・枠の待機理由を確認する。別の担当を同じIssueへ追加するときは、成果PRがopenでも新しい名前とworktreeを指定できる。複数人になったら、既存担当への操作にも `--worker primary` または補助担当名を付ける。
- 手動runの登録を `inspect-worker` で確認したら、`wait-worker --child 1217 --worker quality` を会話のシェルツールで背景実行する。待機コマンド自体はシェルから切り離さず、終了通知を受け取る。`wait_state=ended` は終了・回収の確認。runの成否は `runner_summary`、実際に得た結果・未確認事項は `final_file` と成果物で確認する。待機が終了コード124なら待機時間の満了で、同じrunをもう一度待てる。
- 調査だけを任せるターンは `--planning`。結果から次の方針を決め、必要に応じて `decide-worker-plan` で回答を結び付ける。同じ担当に続きやmajor修正を頼むときは、終了・回収を照合して手動の `run-worker --resume` または自律の `continue-worker` を使う。
- 補助担当はcommitまたは測定結果を返す。統合担当が競合を解消して統合HEADを検証し、担当Issueを閉じるPRに `Closes #1217` を書く。親はPRのbranch・HEADと条件ごとの証拠を確かめ、[正式レビュー](review.md) の `review-request` へ一度渡す。個別に受入だけを記録する場合は `record-result`。
- 複数担当の一人を止めるには `stop-worker --worker 名前`。Issue全体の保留は `stop` を使う。自律親で担当が一人のときの `stop-worker` は、対象Issueを保留し、依存して保留されたjobにも停止を要求する。影響先は `launch_plan` で確認する。停止要求だけでは終了・枠解放は確定しない。[終了が不明な場合](recovery.md)

**未マージの依存PRを土台にする場合**は、`stacked_on` の全headを取り込み、実際に積んだ境界commitを保存する。依存先が一つなら先行branchを後続PRのbaseにする。先行PRのマージ後、cleanな後続worktreeでその境界から後続のcommitだけをmainへ載せ替え、PRのbase・差分・CIを新HEADで確認する。先行がsquash mergeされても、保存した境界を使える。
