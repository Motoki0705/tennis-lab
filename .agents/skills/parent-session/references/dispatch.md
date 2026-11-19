# dispatcherから起動された親

末尾の `bootstrap_input` は予約JSONのファイルパス。中の `parent`、`repo`、`parent_state_dir`、`dispatcher_state_dir`、候補と根拠を読む。以下の変数はこの入力から取る。**dispatcherの `--state-dir` とparentの `--state-dir` は別の場所を指す。**

```bash
export PYTHONPATH=.agents/ops
.venv/bin/python -m dispatcher --parent-state-dir "$parent_state_dir" --state-dir "$dispatcher_state_dir" accept --input "$bootstrap_input"
p=(.venv/bin/python -m parent --repo "$repo" --parent "$parent" --state-dir "$parent_state_dir" --dispatcher-state-dir "$dispatcher_state_dir")
"${p[@]}" status
```

受理した予約の自律根拠を使う。親・子のopen状態、正式な依存、既存PRと保存済み委任を照合し、`launch_plan` から作業を選ぶ。`start-worker` が返す `job_id` と `delegation_id` は予約の証拠。実際の起動はsupervisorに任せ、終了イベントが来たら `inspect-worker` の結果と成果物を読む。[作業の分担と統合](workers.md)

| 現在の状態 | 親が返すもの |
|---|---|
| CI・依存・人間判断を待つ | 独立して進められる仕事を進め、残りは `workflow-report` の `waiting` に理由・再開条件・確認済みの証拠を書く。 |
| 実行や検証が失敗し、進められない | `failed` に失敗した対象、影響、再開に必要な条件を記す。元のjob・runを保持して [復旧](recovery.md) へ進む。 |
| 全条件を受け入れた | [完了通知とfinish](finish.md) を揃えて `workflow-report` の `completed` へ渡す。supervisorが終了を照合する。 |

どの状態でも、最終文は `workflow-report` が返した4キー（`schema_version`, `state`, `reason`, `resume_condition`）のJSONをそのまま返す。

単独Issueがマージで閉じた直後に親が中断したら、同じstateでdispatcherの `resume --parent 親番号 --reason-file resume.md`。マージ・close・後確認の記録を使い、残る全体受入と報告を完了する。子を取り消した場合は、その理由・残作業の引継ぎ先・ワーカー終了を確認して `record-cancellation` に残す。
