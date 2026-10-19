---
name: deepseek-harness
description: オーケストレーター（Codex / Claude Code の親）専用。dshの中から使わない。共通wrapperでDeepSeek Harnessへ実装を委譲し、親が差分・テスト・推定費用を確認して取り込むときに使う。
---

# DeepSeek Harnessへの実装委譲

**オーケストレーター（Codex / Claude Code の親）専用。dshの中から使わない。**
dshとしてこのskillを見つけた場合は、渡された実装タスクだけを進める。
再委譲や、wrapperの再帰拒否を環境変数の解除で回避してはならない。

本体はこのディレクトリだけに置く。Claude Codeは既存の `.claude/skills` symlinkで参照する。
実行の正本は [tools/dsh/README.md](../../../tools/dsh/README.md)。
引数・記録形式・固定モデルと安全設定はそこを読み、このskillでは親の判断を扱う。
実タスクは必ず共通wrapperを通す。既存のレビュー・merge運用と外部書き込みの権限は維持する。

## 導入と前提を確認する

1. 元issueの本文・合意コメント・先行依存・既存担当/PRを確認する。
   未合意の仕様、入力や素材の選定、他の担当の範囲は親が解決してから渡す。
2. runtimeを所有するworktreeのrootで `node tools/dsh/runtime.mjs check` を実行する。
   未導入・版不一致なら [導入手順](../../../tools/dsh/README.md#導入) に従う。
   globalのdsh等へ切り替えない。checkはキーも実APIも不要。
3. 実API利用の許可とキー設定の申告を確認する。未設定、またはwrapperがキーの欠落・不備を
   報告した場合は、[専用キーファイルの設定方法](../../../tools/dsh/README.md#apiキー-専用ファイルだけを読む)
   を示してユーザー自身に設定を依頼し、設定完了の返答まで実行を止める。
   親はキーファイルを作成・探索・読み取りせず、キー値を質問・転記・exportしない。
   設定済みなら検証と読み取りはwrapperに任せる。拒否された設定を自動修正しない。

## 1回で終わる実装単位にする

- 1つの観測可能な変更と、その受入条件・CPUテストを1回のdsh実行へ渡す。
  親が差分とテストで合否を決められ、有限のtimeout内に実装・検証できる大きさにする。
- 複数の独立した機能、先行リファクタ、データ選定、export、描画は分ける。
  例: 配置trace抽出と同値性テスト → bundle exporter → Blender shot。
- ファイルの所有範囲が重なるタスクは同時に実装させない。各単位に専用worktreeを用意し、
  次の単位は親が合格とした先行差分を含むbaseから始める。
- 機密情報を含めずに [task-template.md](task-template.md) を埋める。
  許可パス、対象外、具体的なテストコマンド、必要な入力の参照を確定する。
  テンプレートの未記入欄や未解決の前提が残る状態で実行しない。
  GPU処理・実素材の確認が必要なら別工程とし、その実行条件を親が管理する。

## 専用worktreeとtask.mdを作る

次は親のworktreeのrootで使う例。親のHEADに先行依存が揃っていることを先に確認し、
`task_name` は実装する機能名に変える。メインrepo直下では実装しない。

```bash
orchestrator_root="$PWD"
repo_main="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"
task_name=body-placement-trace
base_ref="$(git rev-parse HEAD)"
worktree="$repo_main/.claude/worktrees/$task_name"
git -C "$repo_main" worktree add -b "dsh/$task_name" "$worktree" "$base_ref"
ln -s "$repo_main/.venv" "$worktree/.venv"
umask 077
task_dir="$worktree/.dsh-runs/tasks/$task_name"
mkdir -p -m 700 "$task_dir"
cp "$orchestrator_root/.agents/skills/deepseek-harness/task-template.md" "$task_dir/task.md"
```

既存のworktreeを再利用する場合は担当・branch・HEAD・既存差分を確認し、別作業へ流用しない。
`.venv` が既にある場合は参照先を確認し、上書きしない。
実行前の `git status --short --untracked-files=all` とHEADを親の記録に残す。
共有 `.venv` や `data/`・`ckpt/` のsymlink先は読み取りだけとする。

## wrapperで実行する

task.mdを完成させ、妥当な正の有限timeoutを決める。例の600秒はタスクに合わせて選び直す。
runnerは導入を確認したworktreeのものを絶対パスで指定する。

```bash
runner="$orchestrator_root/tools/dsh/run.sh"
runner_status=0
"$runner" --task "$task_dir/task.md" --worktree "$worktree" \
  --mode workspace-write --timeout 600 > "$task_dir/initial-summary.json" || runner_status=$?
```

調査だけなら `--mode read-only` を選ぶ。
利用できる引数と拒否条件は [共通wrapper](../../../tools/dsh/README.md#共通wrapper-実行継続停止)、
要約JSONと終了値は [出力と終了値](../../../tools/dsh/README.md#出力と終了値) を参照する。
親の要約JSONはローカルのtaskディレクトリに保存し、wrapper自身の証拠はJSONの `records_dir` から辿る。
実行したtask.md・修正依頼は上書きせず保存し、metadataのtask hashと対応付ける。

## 親が検証して、必要な修正を戻す

1. shellの終了値と要約JSONの `exit_code`・`result`・`turn_end_reason` を照合する。
   空/不正JSONや非ゼロ終了は失敗として原因を確認する。**exit 0 / completedはターンの終了であり、
   タスク成功ではない。** `task_success: null` を合格に読み替えない。
2. `session_id` と記録先を保存し、親自身で `final_path`、イベント、stderrを読む。
   ツール失敗、権限拒否、途中切れ、未実行のテスト、未解決点を確認する。
3. 親自身で実worktreeの `git status`、`git diff HEAD`、未追跡ファイルの内容を確認する。
   `diff_summary` だけで合否を決めず、実行前の記録とも比較して範囲外の変更がないか確かめる。
4. 親がworktreeのrootで指定テストを実行し、受入条件を確認する。
   Pythonは `.venv/bin/python`、pytestは `-n 4` 以下とし、複数のpytestを同時実行しない。
   GPU処理や必要な実データ・動画による検証は、タスクで定めた別工程として親が行う。

不合格なら、失敗した条件・再現コマンド・期待値・修正許可範囲を `fix-1.md` に書き、
**同じworktree・mode・session_id** で修正依頼する。timeoutや中断後は記録と差分を先に確認する。
継続の制約は [永続homeと継続session](../../../tools/dsh/README.md#永続homeと継続session) に従う。

```bash
# session_idには初回summaryから得た値を設定する。
runner_status=0
"$runner" --task "$task_dir/fix-1.md" --worktree "$worktree" \
  --mode workspace-write --timeout 600 --session-id "$session_id" \
  > "$task_dir/fix-1-summary.json" || runner_status=$?
```

**修正依頼は最大2回、初回を含めwrapperの起動試行は1タスク最大3回。**
失敗・中断・拒否された試行も数え、sessionやworktreeを変えて回数をリセットしない。
継続可能なsession_idが得られない、継続が拒否される、前提/範囲の変更が必要、または上限に達したら
自動実行を止め、証拠と未解決点を残して親が引き取るか人間に相談する。
各修正後も親が同じ検証を行う。

## 合格した差分と費用記録を取り込む

親が確認した差分だけを選んでstageし、pre-commitを通してcommitする。
PR作成は親の権限・既存運用に従う。dsh自身にはcommit / push / gh書き込みをさせない。
ローカルのraw記録をcommitしたり、そのままissue/PRへ転載したりしない。

タスクごとに親のローカル記録へ以下を残し、公開する要約は秘密情報・ローカル絶対パスを除く。

- taskの参照、worktree/branch/base、run/session/turn、試行数・所要時間、親の合否と取り込みcommit/PR。
- 親が読んだ差分、実行したテストと結果、各修正の理由・残課題。
- 各runのusage・推定額・通貨、単価の参照元と確認日・計算根拠、欠測と未計上項目。

使用量と推定費用は [費用の契約](../../../tools/dsh/README.md#使用量と推定費用) と
[run/sessionの区切り](../../../tools/dsh/README.md#ローカル記録と使用量の区切り) に従い、
各runの `usage.json` を根拠にする。実API実行前にREADMEの手順で公式単価を照合する。
run合計とsession合計を重複加算せず、失敗した試行も含める。
欠測や `null` を0円にせず、推定範囲と不完全性をそのまま記録する。
**推定額は請求確定額と区別する。**
