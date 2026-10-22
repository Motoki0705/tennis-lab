# 親セッションの担当・進捗・レビュー要求

人間が会話で指定した親issueを1セッションで統括するための明示CLI。
会話の解釈・質問・委任・完了確認は [parent-session skill](../../skills/parent-session/SKILL.md)、
親子issueの書式・ラベル・段階移行は [issues](../issues/README.md) が正本。
このCLIは agent・レビュー・CI・merge を実行しない。timer・poll・hookへの登録も行わない。

Herdrでの担当表示は[herdr_labels](../herdr_labels/README.md)へ連携する。
担当開始・再開・引継ぎ・終了後の`display_sync`で反映結果を確認できる。

## 責務と状態の正本

| 対象 | 正本 |
|---|---|
| 親issueと担当セッションの対応、解除・終了・引継ぎ待ち、委任の起動予約、worktree | マシンローカルの `registry.json` |
| 担当開始、要件確認、要件合意と人間の確認の引用 | 親issueコメント |
| 正式な親子関係、子のopen/closedと理由 | GitHub sub-issue / issue |
| 子の具体的な停止理由・根拠・再開条件・次の対応、改善候補 | 子issue本文 |
| 子の一覧と全体進捗、停止/再開通知、レビュー対象・指示・状態 | 親issue本文・コメント |
| レビュー対象のバイト列 | ローカル `reviews/<run ID>/pr-<N>.diff` と `request.json` |

stateは `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/parent-sessions/`。
同じホストの全worktree・Claude/Codexで共有し、state内のflockで登録・起票・更新を直列化する。
`registry.json` はversion=1、`parents` のキーは `owner/repo#親番号`。
キーの `owner/repo` は大文字小文字を区別せず、小文字に正規化する。
既存キーも読込時に正規化し、同じ親の別名レコードが衝突した場合は、統合・上書きせず読込を拒否する。
各行に `session`（実行環境の識別子）、`status`、`generation`、`tasks`（子番号→担当/branch/worktree/依存/委任予約）、
`reviews` を持つ。ファイルは0600、ディレクトリは0700で作り、fsyncとrenameで保存する。
破損・未知version・取得失敗を空の状態に置き換えない。

識別子は実行agentの環境から取得する。Claudeは `CLAUDE_CODE_SESSION_ID`、Codexは `CODEX_SESSION_ID`。
欠落・空値・両方の混在はエラーにする。人間にIDを聞いたり、引数・PID・作業パスで補ったりしない。
子agentから担当親を装って登録せず、担当する親セッション自身が実行する。
Codexの `CODEX_THREAD_ID` は現在のthread、`CODEX_SESSION_ID` は親子で共有するroot session。
両者が設定されていて異なる場合はsubagentからの操作として拒否し、thread IDを代用しない。
根拠はCodex CLI 0.163.0の公式 [exec_env.rs](https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/core/src/exec_env.rs#L27-L46) と
[process_manager.rs](https://github.com/openai/codex/blob/rust-v0.163.0/codex-rs/core/src/unified_exec/process_manager.rs#L1414-L1428)。
環境を提供しない起動経路では、この入口を使用できない。
セッションIDはissue・コメント・memoryへ出さず、CLIの通常出力にも含めない。
worktreeの絶対パスと再起動証拠はローカル出力にのみ含まれるため、それを公開コメントへ貼らない。

1セッションのactive/引継ぎ待ち親は1件。同じ親の別セッションによる登録も拒否する。
終了をプロセス一覧などから推測せず、親自身の `release --ended` で記録する。
別セッションへの移管は `handoff --accept` による明示操作だけを許可する。
マシンをまたぐ分散排他やセッションの自動選定は範囲外。

## 入口と合意

以下のコマンドは会話を受けたagentが実行する。人間に実行させる手順ではない。
共通引数はサブコマンドの前に置く。`--repo` 省略時は `gh repo view` で特定する。
`--parent` は番号、`#番号`、同じrepoのHTTPS issue URLを受け取る。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 register
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent https://github.com/Motoki0705/tennis-lab/issues/1216 status
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 checking
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 agree --spec tmp/agreement.json
```

`agreement.json` は `purpose` / `scope` / `behavior` / `acceptance` / `human_confirmation` の
5キーだけを持つJSON object。値はすべて非空の文章とし、最後の値に人間の確認発言を引用する。
`checking` → 会話で認識合わせ・合意 → `agree` の順。担当指定だけでは「担当指定のみ」、
`checking` で「要件確認中」、有効な合意コメントで「要件合意済み」となる。
再度 `checking` に戻したら、新しい合意の記録まで分割・委任を拒否する。
会話に実際の合意があるかを判断する責任は親agentにあり、CLIは人間の確認を捏造しない。
既存issueにマーカーなしの人間との合意がある場合は、その発言と合意内容を読み、
`checking` / `agree` で同じ内容を記録する。記録形式を移すためだけの再承認は求めない。

コメントの1行目は `<!-- parent-session:v1 <JSON> -->`。
JSONは `parent`、`event` とイベント固有の値を持ち、`<` / `>` をUnicode escapeする。
現在の `gh api user` と同じloginによるコメントだけを読み、本文中の引用マーカーを無視する。
合意には構造化した `agreement` と、そのcanonical JSONの `agreement_sha256` を含める。
canonical JSONはUTF-8、`ensure_ascii=False, sort_keys=True, separators=(",", ":")`。
再実行時は同一イベントを再投稿しない。別ghアカウントへの移管は自動で信用せず、
新しい担当アカウントで改めて要件確認・合意を記録する。

`create-child` と `delegate` は、ローカル担当がactiveであり、親がopenであり、
親コメントの最新段階が合意済みであることを**書き込み前**に検査する。
合意後の分割結果への追加承認は設けない。起票元が人間・レポートのどちらでも登録入口は同じ。
単独で完結する既存issueは登録・合意してそのまま扱い、不要な親子階層を作らない。
分割するときだけ、issues規約に従ってそのissueを親へ移行する。

## 採用提案の受領

日次triageは指定親のactiveな登録を確認し、親コメントに `event: proposal_pending` を記録する。
担当親は自分のセッションから次を実行する（triageの再実行を待たず同じ入口から受領することも可能）。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 receive-proposal --report 1300 --proposal R-20261019-01
# 会話でその提案の要件を認識合わせし、人間の合意を得た後
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 agree --report 1300 --proposal R-20261019-01 --spec tmp/proposal-agreement.json
```

両イベントは `source: {proposal_id, report_number, option}`、`adoption: selected` と
`proposal_sha256` を持ち、セッションIDを公開しない。digestは提案の本文・選択肢の定義に結び付け、
変更可能な引渡し表示と関連親の指定を含めない。受領済みの再送は冪等。
受領は実装への同意ではなく、その提案だけを要件確認待ちにする。親全体のphase・合意と、
既存の独立作業・レビュー証拠は維持する。親自体が合意済みの状態で、上記の提案指定付き `agree` に
同じ5キー形式の合意specを渡す。`event: proposal_agreed` に受領時のsource・提案digestと合意内容を記録し、
その提案由来のcreate-child/delegateだけを許可する。他の受領提案や既存の親合意を流用しない。
statusの `proposal_receipts` / `proposal_agreements` で未合意と合意済みを照合でき、受領直後の中断も
GitHubコメントから復元する。明示的な `checking` は親全体と提案別の合意を取り消すため、
親の `agree` と各提案の合意を再記録するまで、それぞれの実装を拒否する。
既存提案の合意内容を変更する場合も `checking` で明示的に取り消してから記録する。
未取消の合意は同じ内容の再送だけを許可し、レビュー後に対象作業の要件だけを黙って差し替えない。
親未指定・未登録・停止中なら受領できない。人間の指定/再開後に親自身がregister/handoffして同じ入口へ戻る。
採用チェック・選択肢・提案本文が後から変われば起票/委任時も拒否し、担当親が採用と要件を再整理する。
受領・起票・委任時は、出典がopenで完了済みの `ops-report` issueであることも検査する。
人間が独立に作ったissueには受領操作は不要で、同じregister/checking/agreeから開始する。

## 子の作成・委任・再開

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 create-child --spec tmp/child.json
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 reconcile
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 delegate --child 1217 --agent codex --branch codex/parent-session --worktree /path/to/main/.claude/worktrees/parent-session
```

子manifestは `{"key":"parent-session", "spec":{...}, "dependencies":[1225]}`。
`spec` は [issues renderの子spec](../issues/README.md#cliと整合検査) をそのまま使う。
親番号・識別名は実際の担当親と一致させる。`key` は同一作業で固定する小文字英数・ハイフンの1〜64文字。
`dependencies` 省略時は子本文の依存欄の `#番号` を先行依存として扱う。
明示した場合はその値で依存欄も生成する。依存先は同じ親の既存の直属子に限り、循環は拒否する。
既存の自由文に後続タスク番号が混じる場合は、`delegate --dependencies 番号...` で先行だけを明示する。
進捗判定は常に最新の子本文の依存欄を読む。明示CLI入力は委任時に本文へも反映し、古いローカル記録で上書きしない。

生成本文の末尾に `<!-- parent-task:v1 parent=N key=KEY spec=SHA256 -->` を付ける。
open/closed全issueをページングしてキーを照合し、spec不一致・複数一致を拒否する。
既存があれば再利用し、ラベル・正式なsub-issue・親一覧の不足だけを揃える。
作成応答の喪失や関係登録・親編集の途中失敗も、**同じキーとspec**で再実行する。
別の正式な親へ所属したissueを自動付替えしない。stateの消失だけを理由に新しいキーで作り直さない。
レポート由来specは受領・その提案を含む要件合意・現在の採用を検証し、共通 `index_proposals` でも照合して、既存提案を別キーで二重起票しない。
同じ提案の既存子はキー・タイトル変更・親付替え後も再利用し、現在の正式な親を返す。自動で親を戻さない。
親関係のない既存提案が別キーなら番号を示して拒否し、孤立issueを明示的に移行・照合してから扱う。

`delegate` は実在する専用git worktree・branch・agent prefixを確認し、起動予約を先に保存する。
`action: launch` を受けた親がサブエージェントを起動する。CLI自身は起動しない。
既存PR・委任予約・完了・取消・停止・依存保留があれば起動を拒否する。
`reconcile` は正式な子、全状態のPRの `Closes/Fixes/Resolves #子番号`（同repoのURLも可）、
記録済みbranchを照合して、PRと担当・検証記録を返す。
PRは子番号ごとのclosing指示を明記し、複数子を閉じる場合も各番号に指示を付ける。
ブランチ名だけから親子関係を推測しない。
既存PRや予約がある場合の `do_not_launch` は再開の調査対象であり、成功や完了を意味しない。
予約後の起動失敗・中断時は、既存workerが終了した証拠を確認し、必要なら停止・再開を記録して
`delegate --restart-evidence-file tmp/evidence.md` で明示的に再予約する。既存PRがあれば再予約もしない。

`record-result --child N --evidence-file tmp/tests.md` はclosing指示とbranchを照合し、
子の成果物欄へPRと検証結果を追記する。記録だけで子を完了扱いにせず、過去の検証記録も残す。

## 停止・再開・改善候補

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 stop --child 1217 --spec tmp/stop.json
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 resume --child 1217 --spec tmp/resume.json
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 improvement --child 1217 --spec tmp/improvement.json
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 sync
```

停止・再開specは `reason` / `facts` / `impact` / `resume_condition` / `next_action` の5つの非空文章。
再開specには条件が満たされた証拠を記載する。詳細を子へ残してから親の進捗・短い通知を更新する。
子の停止節末尾に `<!-- parent-task-status:v1 state=stopped|resumed detail=SHA256 -->` を記録し、
途中失敗後もその公開状態から停止判定を復元する。本文編集中に取得内容が変わった場合は上書きせず拒否する。
通知は子へのリンク、理由の要約、依存保留と継続の一覧、次の対応を持ち、事実の詳細は転記しない。
停止・改善候補・検証結果の記録は担当activeなら要件確認中でも行える。記録から実装は起動しない。
進捗は停止→依存先へ保留を伝播し、独立した子は継続する。再開で保留を再計算する。
`sync` はGitHubの実状態・完了理由から一覧を更新するだけで、agentを起動しない。
`closed/completed` だけを完了チェックにし、取消は未チェックで残す。
親全体の完了はチェック数から推測せず、全体の受入条件を親agentが確認する。

改善specは `purpose` / `evidence` / `scope` の3つの非空文章。
子の改善候補欄だけを更新し、Acceptance・実装範囲・担当状態は変えない。

## 担当の終了・引継ぎ

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 handoff --note-file tmp/handoff.md
# 指定された次の親セッション自身が実行する
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 handoff --accept --note-file tmp/takeover.md
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 release --ended --note-file tmp/completion.md
```

`handoff` で引継ぎ待ち、`release` で解除、`release --ended` で終了済み。
引継ぎ待ちは旧担当も分割・委任できず、新担当のacceptまで代替起動しない。
acceptは通常、解除/終了/引継ぎ待ちだけを受け取り、同時担当数を再検査してissue/PRを照合する。
中断した同一セッションは `reconcile`、別セッションは明示acceptから始める。
旧セッションが未解除のまま失われた場合は、ローカルstateと既存workerを確認し、
人間の担当指定に基づき `handoff --accept --recover --note-file tmp/recovery.md` を明示実行する。
ノートに旧workerの停止・既存worktree/PRを確認した証拠を残す。通常のregisterはactive担当を上書きしない。
recoverも既存の委任予約を保持し、照合結果から自動再起動しない。stateを消して空き担当と扱わない。
引継ぎノートには完了/未完了の子・PR、停止条件、次の作業、レビューrunを記載し、ローカルIDを書かない。

## #1218へのレビュー要求契約（schema version 1）

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 review-prepare --prs 1250 1251 1252 > tmp/targets.json
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 review-request --run-id parent-1216-UUID --instruction-file tmp/instruction.md > tmp/review-request.json
```

`review-prepare` は全直属子にclosing指示がある全open PRを対象集合と照合する。
PR未取得・欠落・重複・closed・draft・fork・親と無関係をエラーにし、部分的な要求を公開しない。
未実装の子にPRがまだ無いことは許容する。段階ごとに、その時点の全open子PRを束ねる。
分割していない単独issueではそのissueを閉じるPRを同じ方法で扱う。
GitHub取得・git差分取得後にPR集合とhead/baseを再取得し、途中更新を拒否する。
不足オブジェクトだけ `origin` からSHAをfetchする。

prepareのJSON（追加キーなし）:

```json
{
  "schema_version": 1,
  "review_run_id": "parent-1216-32桁のhex UUID",
  "repo": "Motoki0705/tennis-lab",
  "parent_issue": 1216,
  "reason": "親が対象PR群を確定した理由",
  "prs": [{
    "number": 1250,
    "head_sha": "40桁のhex SHA",
    "base_sha": "GitHubで取得したbase先端の40桁SHA",
    "base_ref": "main",
    "merge_base_sha": "baseとheadのmerge-baseの40桁SHA",
    "diff_sha256": "64桁hex SHA-256",
    "branch": "codex/issue-format"
  }]
}
```

`prs` は番号昇順。diffは以下の**生バイト列**のSHA-256（文字列への変換・切詰めなし）。

```bash
git diff --no-ext-diff --no-textconv --binary --full-index --no-renames --no-color MERGE_BASE HEAD --
```

親コメントの `event:review, state:targets_fixed` にrun ID・requestのcanonical JSON SHA-256を記録し、
本文のJSONコードブロックにmanifest全体を載せる。この時点ではレビュー未指示。
`review-request` は同じ対象が現在も有効なことを再検査して、親の明示指示をコメントへ記録する。
出力は上記manifestに `"review_state":"instructed"` を追加したJSON。
reviewイベントには現在の担当 `generation` と最新合意の `agreement_sha256` も記録する。
公開のmanifestは変更せず、セッションIDを出さずに指示と担当・合意を結び付ける。
`request_sha256` は `review_state` 追加前のprepare manifestのcanonical JSONに対して計算する。
**#1218はこのJSONだけで起動せず、親コメントの同じrun ID/request digest/指示済み記録と最新対象を照合し、
run IDごとの実行を一度だけ受理する責任を持つ。** prepareだけのJSONは実行指示にしない。
受理・実行・修正確認・結果スキーマは [親レビュー入口](../review/README.md) を参照する。
同じprepareの再実行は同じrun ID、同じrequestの再送は同じ指示を使う。
次のrunや対象変更には `review-prepare --reason-file` で新しい理由を必須とする。
PR open/push/CI完了ではこの入口もレビューも起動しない。

未指示 → `targets_fixed`（対象確定）→ `instructed`（指示済み）→ `result_received`（結果受領）→
`fixes_verified`（修正確認）→ `complete`（完了）の順。
後半3段階は `review-state --run-id ID --state 状態 --evidence-file tmp/evidence.md` で証拠を残す。
結果受領は成功判定ではなく、失敗・阻害指摘も受領して記録する。親が指摘と修正を照合し、
各PRのCIなどを確認した証拠が揃うまで完了へ進めない。CLIは#1219のmerge gateの代用ではない。

## 将来の割当側と実行条件

将来の自動割当は「どのセッションにどの親を担当させるか」だけを決め、対象セッション内で
`register` を呼ぶ。同じ入力（repo・親番号/URL）、環境由来ID、state、競合エラーを使い、
合意・子作成・実装・レビューを自動的に飛ばさない。採用提案の入力でも同じ境界を守る。

CLIはJSONをstdout、エラーをstderrへ出し、成功0・設定/取得/整合エラー2で終了する。
ghは現在認証中のアカウント、GitHub書込みにはIssuesのwrite権限が必要。
親子ラベルの作成やsystemd操作は行わない。GitHubとローカルstateは単一トランザクションではないため、
非ゼロ終了後は同じ入力で照合・不足更新を行う。未完了を成功や新規起票に置換しない。
atomic write中にプロセスを強制終了して `.tmp` が残った場合は、稼働中プロセスが無いことと
registryの内容を確認して、その一時ファイルだけを人間の管理下で退避してから再試行する。

テスト: `.venv/bin/python -m pytest tests/e2e/agents_ops/test_parent_session.py -n 4 -q`。
fake gh・一時state・使い捨てgitを使い、本物のCodex/Claude/GitHub書込みは起動しない。
