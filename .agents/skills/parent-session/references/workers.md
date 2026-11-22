# ワーカーの委任と統合

## Issueと担当を決める

「推論を高速化し品質を維持する」は一つの成果。実装と品質比較を別ワーカーに任せ、劣化が見つかれば同じIssueに調査担当を足せる。一方、比較で手法の採否を決めてから本番導入するなら、採否を確定する地点で子Issueを分ける。

正式な直属子があれば `--child` は対象の子番号。上位親も直属子もないIssueでは、自身の番号を指定する。親が直接実装する場合は専用worktreeで変更し、作成したPRに受入条件ごとの証拠を揃える。新しい子を作る場合は、受入条件と先行依存を [子Issueの書式](../../../ops/issues/README.md#本文) で定める。同じ子の作成を再送するときは同じkeyとspecを使う。

## 同じIssueに二人を割り当てる例

親1216・担当Issue1217の例。両担当の基準commitを固定する。`--from` が担当別のbranch・worktree・共有入力・venv・出力を準備し、予約と初回起動を行う。メインcheckoutの `.claude/worktrees/` に作成する。各担当への指示に作業範囲・受入条件・返すcommitまたは測定結果を書く。

Claude Code / Codex の手動親では、`setsid ... &` で背景起動する。

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

### agy手動親での起動と監視

`agy` 手動親では `setsid ... &` を使わない。`agy` の `run_command` はシェル終了時に背景プロセス（`setsid ... &` を含む）を回収するため、ログ 0 バイト・未委任のまま `exit 0` で失敗する。仮に親シェル終了後に残っても、reparent により [agy_session.py](../../../ops/lib/agy_session.py) の前景TUI祖先照合に通らない。

- `run-worker` は `setsid` や末尾 `&` を付けず、`run_command` の非同期タスクとして直接起動する（起動失敗やターン終了の通知も `run_command` で受け取れる）。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 run-worker --child 1217 --worker primary --from "$base" --prompt-file impl.md
```

- コマンド投入やシェルの `exit 0` を起動成功とみなさない。必ず `inspect-worker` で手動runの登録・稼働（`result: "running"` や進行済みの状態）を確認してから、`wait-worker` も同様に `setsid` や末尾 `&` を付けず `run_command` の非同期タスクとして接続する。`--from` のworktree準備中にまだ未登録なら、実行中の `run-worker` タスクの状態を確認し、準備後に `inspect-worker` を再実行する。
- `inspect-worker` で稼働を確認し `wait-worker` を接続するまで（または準備中の確認手段を確保するまで）、監視がない状態でターンを終了しない。`wait-worker` 接続前に `run-worker` が終了して `followup.state` が継続中の場合は、その完了通知で直ちに `wait-worker` を接続する。

## 待つ、追加する、受け入れる

- `launch_plan` の `action=ready`、既存予約、依存・枠の待機理由を確認する。別の担当を同じIssueへ追加するときは、成果PRがopenでも新しい名前とworktreeを指定できる。複数人になったら、既存担当への操作にも `--worker primary` または補助担当名を付ける。
- 手動runの登録を `inspect-worker` で確認したら、`wait-worker --child 1217 --worker quality` を会話のシェルツールで背景実行して終了通知を受ける（`agy` 手動親では `setsid` / `&` を付けず `run_command` の非同期タスクとして直接起動する）。`wait_state=ended` は、宣言を受けた監視と自動再開がある場合はその一連の終了まで含む。最後のrunの成否は `runner_summary`、成果は `final_file`、監視の成否・親の判断・再開回数は `followup` で確認する。終了コード124は読取側の時間満了。同じ処理をもう一度待てる。終了コード2は異常。保存済みの理由を読んで [復旧](recovery.md) へ進む。
- 調査だけを任せるターンは `--planning`。結果から次の方針を決め、必要に応じて `decide-worker-plan` で回答を結び付ける。同じ担当に続きやmajor修正を頼むときは、終了・回収を照合して手動の `run-worker --resume` または自律の `continue-worker` を使う。
- 補助担当はcommitまたは測定結果を返す。統合担当が競合を解消して統合HEADを検証し、担当Issueを閉じるPRに `Closes #1217` を書く。親はPRのbranch・HEADと条件ごとの証拠を確かめ、[レビューとマージ](review.md) の手順に進む。個別に受入だけを記録する場合は `record-result`。
- 複数担当の一人を止めるには `stop-worker --worker 名前`。Issue全体の保留は `stop` を使う。自律親で担当が一人のときの `stop-worker` は、対象Issueを保留し、依存して保留されたjobにも停止を要求する。影響先は `launch_plan` で確認する。停止要求だけでは終了・枠解放は確定しない。[終了が不明な場合](recovery.md)

**未マージの依存PRを土台にする場合**は、`stacked_on` の全headを取り込み、実際に積んだ境界commitを保存する。依存先が一つなら先行branchを後続PRのbaseにする。先行PRのマージ後、cleanな後続worktreeでその境界から後続のcommitだけをmainへ載せ替え、PRのbase・差分・CIを新HEADで確認する。先行がsquash mergeされても、保存した境界を使える。

## ナレッジ記録PRの作成

作業者は、独立した実験・文献の記録と、run同梱の再現専用Python・シェルを、最初から記録専用PRにまとめる。

- 記録だけで受入条件を満たすIssueは、記録PRに `Closes #<issue>` を書く。
- 実装に付随する記録PRは、`Refs #<issue>, #<実装PR>` のように対象Issueと実装PRを参照する。受入条件を満たす実装PRにclosingを置く。
- 実装と混在する差分、Web UI・検査器・スキーマ・レジストリ（`deployments.yaml` 等）・保存規則・CI設定の変更、本番チェックポイントと一体の採用根拠は、通常PRの正式レビューで扱う。

記録専用PRの本文に `## 実行証拠` 見出しを置き、直下に `json` のコードブロックを1つ書く。`head_sha` は検証したPR HEADの完全SHA。run同梱の再現スクリプトを追加・変更した場合や、実行結果が受入条件の場合は、実際のコマンド、終了値、結果を書く。`result` には受入条件に対応する測定値・出力や実行ログの参照を含める。

```json
{
  "head_sha": "0123456789abcdef0123456789abcdef01234567",
  "status": "passed",
  "command": "bash knowledge/runs/example/repro.sh",
  "exit_code": 0,
  "result": "評価32件が完了し、指定された全出力を生成した。"
}
```

実行を伴わない受入では `{"head_sha":"0123456789abcdef0123456789abcdef01234567","status":"not_required"}` を同じ形式で記載する。HEADを更新したら、実行や受入の照合を新HEADで済ませてから、この節を更新する。

## CI・ローカルqueueの待機

[共通のワーカー指示](../../../ops/parent/worker_prompt.md) が、最終メッセージ末尾の `PARENT_RESULT` の形式を定める。親が書く具体指示も、投入／pushまでの作業と、結果を受け取ってからの作業に区切る。

- 良い例「検証してcommit・pushし、PR番号と完全なHEADを宣言して終了。Python成功後にドラフトを解除するよう後続作業に指定する」
- 悪い例「pushしたら、このターンでPython成功を確認してドラフトを解除する」
- 計測の例「条件AとBを共有queueに投入し、両方のjob IDを宣言して終了。結果を受け取ったターンで集計して記録する」

手動委任では `run-worker` が宣言を検査する。監視はユーザーのsystemd transient serviceで実行する。GitHubとCodexのCLI認証は、そのユーザーの設定を使う。CIは宣言したPR・HEADのPythonを照合する。Pythonが未登録またはskipで、Git差分が記録専用と確定した場合は [Knowledge library の結果](../../../ops/lib/README.md#ナレッジ記録専用prのci) を照合する。失敗したときと、成功後の後続作業があるときは同じsessionを再開する。queueは宣言した全jobの終端を待ち、終了状態・ログ・取得できた出力先をまとめて一度だけ渡す。再開が必要で枠が埋まっているときは、CLIが結果を保存し、空き枠を取得して再開する。再開は担当ごとに3回まで。上限や人間の判断が返ったら、理由と成果を照合する。

共有queueの `worker: RUNNING` は親側が委任前に確認する。停止中なら親側の独立したサービスで `training_queue.sh serve` を起動する。委任した子はworktreeから `add` を行い、queue workerを起動しない。子の共通ランナーは終了時に子孫を回収するため、子から起動したqueue workerと実行中jobも停止する。終了値143では、子の終了とqueue workerの停止が同時刻か確認する。専用のqueueを使うなら、親CLIの `TRAINING_QUEUE_DIR` を起動前に指定して子と共有する。

自律親にも同じ宣言が届く。子の終了イベントで `inspect-worker` と `final_file` を照合し、外部の結果を得た後に `continue-worker` で継続する。[自律親](dispatch.md) の待機手順を使う。自動CI・gate修正は専用JSONを受ける [レビュー進行](review.md) がCIを監視する。

## Colabの長時間ジョブ

起動と異常終了・VM消失後の再開は、共通ランナー経由でCodex子（既定のmodel・effort）へ委任する。親は委任時に安定条件を指定する。子は最初の周期checkpointのDrive保存など、指定された条件を確認して終了する。その後は親が完了・異常終了・プロセスやVMの終了をイベントで監視する。環境準備と復旧は [tennis-colab](../../tennis-colab/SKILL.md) に従う。
