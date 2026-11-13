# 親セッションの担当・進捗・レビュー要求

親issueを1セッションで統括するための明示CLI。人間合意と自律着手判断を別の実行根拠として扱う。
会話の解釈・質問・委任・完了確認は [parent-session skill](../../skills/parent-session/SKILL.md)、
親子issueの書式・ラベル・段階移行は [issues](../issues/README.md) が正本。
`start-worker` / `continue-worker` は委任予約をdispatcherへ渡す。実agentの起動はsupervisor、
レビュー・mergeは親が明示する既存入口で行う。timer・poll・hookへの登録は行わない。
手動親は [run-worker](#手動workerの起動継続とrun記録) で委任と共通ランナーのrunを結び付けて起動する。
自律親の開発手順とworker/reportコマンドの契約は [development workflow](development_workflow.md) が正本。

Herdrでの担当表示は[herdr_labels](../herdr_labels/README.md)へ連携する。
担当開始・再開・引継ぎ・終了後の`display_sync`で反映結果を確認できる。
表示する要約（`display_summary`）は親ごとの記録に保存し、`register --summary`と`summary --text`で設定する。
幅・検証・要約がない場合の扱いはherdr_labelsの[要約](../herdr_labels/README.md#要約)が正本。
同じ位置で[board](../board/README.md)の名前も同期し、結果を`board_sync`へ返す。

## 判断と機械的な検査

目的・範囲・受入条件・権限への影響と、既存証拠の適用可能性は担当親が判断する。
CLIは現在の担当、対象repo/issue/PR/branch、runとsession、終了と回収、保存と公開の実状態を照合する。
ハッシュは記録・対象バイト列の同一性に使い、自由文の意味や計画の妥当性の代わりにしない。

`status.input_changes` は合意時からの本文変化とコメントの追加・編集・削除の参照を返す。
親は変更を読み、合意の範囲内なら進め、実質変更なら必要な範囲を確認する。読み取りでは来歴を改変しない。
`checking` は確認中の表示。既存の合意は明示的な `basis-revoke` まで残る。`agree` の前にcheckingを強制しない。
子だけの停止は `stop`、ログ保留は公開、merge条件不足はそのmergeへ限定する。

レビューの出典は作成時のrequest・実行記録・公開結果として保持する。
担当や合意revisionの変化でその証拠を捨てず、親が現在の対象との関係を確認する。
実際のコード差分・CI・指摘解消はreview/merge入口が引き続き照合する。

## 責務と状態の正本

| 対象 | 正本 |
|---|---|
| 親issueと担当セッションの対応、解除・終了・引継ぎ待ち、委任の起動予約、worktree | マシンローカルの `registry.json` と `parents/` |
| 担当開始、要件確認、要件合意と人間の確認の引用 | 証拠用Issue（旧親コメントも読取） |
| 自律着手の根拠・入力・コメント分類・取消 | 証拠用Issueの `execution_basis` / `basis_revoked` イベント |
| 正式な親子関係、子のopen/closedと理由 | GitHub sub-issue / issue |
| 子の具体的な停止理由・根拠・再開条件・次の対応 | 子issue本文 |
| 改善候補・切り出し判断・後継への人間判断 | 元親の証拠用Issueの構造化コメント（旧親コメントも読取）。子の欄は参照 |
| 子の一覧と全体進捗・判断待ち・成果/残件 | 親issue本文の進捗概要（証拠・通知へのリンク） |
| レビュー対象・指示・状態・結果・修正確認 | 証拠用Issue、各PRのprojection/followup |
| 親AIが意味を判断して選んだ人間向け通知 | 親Issueの `notification` と証拠用Issueの `notification_record` |
| レビュー対象のバイト列 | ローカル `reviews/<run ID>/pr-<N>.diff` と `request.json` |

stateは `${XDG_STATE_HOME:-~/.local/state}/tennis-lab-agents/parent-sessions/`。
同じホストの全worktree・Claude/Codexで共有する。更新操作は親ごとの操作ロックを先に取り、
registryロックで担当の衝突・二重予約を防ぐ。GitHub通信中はregistryロックを離す。
取得前後で対象親の担当・generation・tasks・reviewsが変われば終了値2で停止する。
競合やregistryの再検証失敗後は、そのトランザクションでのsave・移行・再取得を拒否する。
呼出元が例外を捕捉しても古いregistryを保存できず、新しいトランザクションで読み直す必要がある。
他の親だけの変更は、その親の保存済み行を取り込み、自分の未保存の変更と参照を保持して続行する。
`status` / `reconcile` / `basis-input` / `basis-check`は操作ロックを取らない。
読取り入口とdry-runはStore.snapshotで検証し、registry lockのファイルも作らない。
GitHub GETの再利用は1回の読取りCLI内に限定し、registryに束縛しない通常readerの鮮度検査は変えない。
Project書込み操作は親の操作ロック→Project共通の操作ロック→registryロックの順に取る。
途中まで保存・公開済みの操作は取り消さず、非ゼロ終了後は不足分だけ再実行する。

状態のキーは小文字の `owner/repo#親番号`。version=1も読み書きできる。
version=2では `registry.json` に担当・status・generationの最小索引だけを持ち、
`parents/` 内の親ごとのJSONにtasks・reviews・拡張フィールドを保存する。
通常の子・レビュー更新では他の親のファイルも索引も書き換えない。
ファイル0600・ディレクトリ0700、fsyncとrenameで保存する。
破損・重複キー・別名衝突・未知version・索引との不一致を空の状態に置き換えない。
旧公開記録、実行根拠、委任予約、レビュー証拠は書き換えない。

### 状態の移行

全親のCLI呼出元をこの版へ更新してから、次を明示実行する。
旧worktreeを更新できない場合は、mainの `.agents/ops` を `PYTHONPATH` に指定して新CLIを使う。
子workerのコード・会話や学習ジョブを停止する必要はない。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1483 state-migrate
```

元の全記録を `registry-v1.backup.json` に保存し、親別ファイルを作成した後、索引をversion=2へ切り替える。
既存担当・世代・委任・レビュー・拡張フィールドをすべて保持する。再送は冪等。
移行後に旧CLIを使うと未知versionで停止する。旧CLIへ戻すためにversionだけを書き換えてはならない。
移行中の旧CLIは取得前後の比較で停止する。失敗時は新CLIで保存済み状態を照合して再実行する。
移行前のbackupは移行後の更新を含まないため、復旧では新しい親別記録と照合する。

### 取得範囲と計測

親の家族に属するissueのtimelineからclosing指示を持つPRを発見し、番号指定で取得する。
親CLIの1操作でrepo全体のPR一覧を列挙しない。コマンド内のGETは再利用し、書込み対象を失効する。
sub-issue一覧に含まれる完全なissue本文も再利用する。固定snapshotの再照合と書込み直前の本文は再取得する。
ページ内・ページ間の重複は最大3回取り直し、解消しなければ失敗する。空結果への変換はしない。

`PARENT_METRICS_FILE` にローカルJSONLの保存先を指定すると、コマンド・経過秒・lock待機秒・
GitHub要求数/ページ数/通信秒・cache hit・処理秒・競合数・終了値を記録する。
再実行時は `PARENT_METRICS_ATTEMPT=2` などで試行番号を渡し、retry_countを記録できる。
本文・取得URL・識別子・資格情報は記録しない。未指定ならファイルを作らない。

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
`--state-dir` でdispatcher/review/mergeと同じregistryを明示できる。省略時は従来のXDG state。
workerの起動・継続では `--dispatcher-state-dir` も同じscheduleへ向ける。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 register --summary 親担当の要約表示
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 summary --text 要約表示の実機確認
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent https://github.com/Motoki0705/tennis-lab/issues/1216 status
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 checking
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 agree --spec tmp/agreement.json
```

`agreement.json` は `purpose` / `scope` / `behavior` / `acceptance` / `human_confirmation` の
5キーだけを持つJSON object。値はすべて非空の文章とし、最後の値に人間の確認発言を引用する。
確認が必要なら `checking` で表示し、人間との合意を `agree` に記録する。既に明確な合意があれば直接 `agree` へ進める。担当指定だけでは「担当指定のみ」、
`checking` で「要件確認中」、有効な合意コメントで「要件合意済み」となる。
再度 `checking` に戻しても、既存の親合意と提案別の合意は維持する。停止・取消は `basis-revoke` で明示する。
`checking` / `agreed` のrevisionは両イベントの過去最大値から進め、取消の連番とは分ける。
取消後の `checking` は新しいrevisionを作り、人間が同じ内容へ再合意しても新しい `agreed` を残す。
取消による要件確認中の状態から直接 `agree` で実際の再合意を記録する場合も、新しいrevisionを使う。
同じ確認中の `checking` と、有効な同一合意への `agree` の再送は冪等で、新しい合意に数えない。
会話に実際の合意があるかを判断する責任は親agentにあり、CLIは人間の確認を捏造しない。
既存issueにマーカーなしの人間との合意がある場合は、その発言と合意内容を読み、
`agree` で同じ内容を記録する。記録形式を移すためだけの再承認は求めない。

### 補足追加の確定待ち

`checking` と `agree` は[共有の補足追加判定](../issues/README.md#親issueの補足追加状態)が
`pending` なら、コメントなどを書き込む前に拒否する。待機には次の読み取り専用入口を使う。
担当登録・session identityは不要で、registryや親の操作lockを保持したまま待たない。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 wait-supplement
```

workflowファイル名は `issue-supplement.yml`、runの表示名は厳密に `issue-supplement #<親番号>`。
ページを辿って一致する最新runを探し、`gh run watch <ID> --repo <repo>` で終了を待ってから
GitHubのラベルを再取得する。定期的にrunやラベルを確認するループは持たない。
JSONは `{parent, supplement_state, run_id}`（既に確定または非対象なら `run_id=null`）。watchの表示はstderrへ出す。
runが無いまま保留なら `workflow_dispatch` の手動実行、または人間によるskipの付与を案内する。
runの終了後も保留なら成功扱いにせず、runの確認と再実行を案内する。待機コマンド自身はラベルを変更しない。

本文更新の内部API `update_body(n, body, expected_body)` は計算元の本文を必須にする。
PATCH直前に `require_final` が書込み対象 `n` をキャッシュを使わず1回取得し、確認したissueを返す。
`update_body` はその同じ応答の本文を照合する。補足追加がpendingなら拒否し、確定後でも本文が
元本文とも更新先とも異なれば上書きしない。本文とラベルを別のGETで確認しない。
既に更新先と一致する確定済み・非対象issueへの再送は書き込まない。

呼び出し元の監査: 子作成・委任・取消記録は合意または実行根拠を要する。一方、`sync_progress`
（`sync`・委任上限変更・worker終了記録からも呼ぶ）、停止/再開・改善候補・確認中の成果記録は、
既存仕様として担当activeだけで行える。これらも共通の `update_body` を通り、pending中の対象本文を書かない。
合意前の事実記録を禁止する変更はしていない。

補足追加側はpendingの間だけ本文を書き、`update_body` は同じ応答で本文の一致と確定状態を
確認してから書く。このため、通常の初回補足追加とは直列になる。本文を計算した後に補足追加が
完了していれば、確定ラベルがあっても本文の不一致で止まる。合意を要する操作では
`checking`・`agree` のgateも通す。ただし最終確認からPATCH送信までの窓はatomicではなく、
通信遅延にも左右される。この窓で手動再実行が確定ラベルを外してpendingへ戻し、本文へ補足すると、
すでに走っている書込みと重なる場合が残る。
手動再実行時は担当親の本文更新を止めてから行い、競合を疑う場合は編集履歴と現在の本文を照合する。
triageの履歴検査と再適用は [ops-report README](../ops-report/README.md#採用提案の自動issue化) を参照する。

### 機械証拠の保存・共通reader

機械記録は `GitHub.record_history(parent, publisher)` が返す `parent.history.History` を通して読む。
従来の親コメントと、親ごとに1件の証拠用Issueのコメントを同時に取得・検証する。
`History.records` は型付きの記録で、元payload・原文・`source`（`legacy_parent_comment` /
`evidence_issue`）・comment ID/URL・publisher・`publisher_verified` を保持する。
`session_events()` は従来のpolicy入力、`comments` はレビュー等の既存parser向けの投影。
単一媒体を読む `GitHub.comments()` はPRの投影や一般の資料取得用に残す。
`policy.events`・`parent_review_policy.records`・`judgement_comments` は媒体を選ばない純粋validatorとして再利用する。

証拠用Issueの取得は `parent evidence`、明示作成は `parent evidence --create`。
どちらも `--repo`・`--parent`・`--state-dir` は通常のCLIと共通。作成の前に管理者が
[ラベルの同期CLI](../issues/README.md#ラベル)で専用ラベルを準備する。
新しい機械記録の初回書込みが `ensure` を呼び、以後は同じ証拠用Issueへ保存する。
旧コメントをコピーせず、既存の同一記録はその実URLを再利用する。

証拠用Issueの本文1行目は次のJSONマーカー。現在の認証publisherが作者であり、同じrepository内で、
ラベルが `parent-evidence` だけであることを毎回GitHub応答で照合する。タイトルは結び付きの根拠にしない。

```html
<!-- parent-evidence:v1 {"parent": 123, "repo": "owner/repo"} -->
```

`repo` はcasefold、`parent` は正整数。マーカーの追加キー・重複JSONキー・未知version・別publisher・
ラベル不一致を拒否する。ローカル番号だけでは信用しない。検索indexを使わず、全状態のIssue一覧を
全ページ取得して候補を検査し、複数候補があれば選ばずエラーにする。
dispatcherの1回のread-only収集内では発見用の一覧を共有し、次の収集で再取得する。
通常はopenのまま保持する。終了による完了数の増加や再開時の状態変更を避け、追記する保存容器であることを示すため。
closedにしても履歴は読め、作業対象への除外は状態に依存しない。
作業Issueへの誤用を防ぐ共通判定と経路は [issues API](../issues/README.md#証拠用issueの除外) を参照する。

作成は共有state配下 `evidence-issues/` の親・repo別lockで直列化する。
このlockを待つ間は親registryを解放し、先行作成のGitHub I/O後の再取得を妨げない。
POST前に送信intentをfsyncして保存し、応答とGitHub再取得を照合してから番号を保存する。
応答喪失・中断後は同じ入口で既存Issueを照合する。送信済みで候補が見えない場合は
`creation uncertain` として止め、再POSTしない。GitHubのIssue作成APIに冪等keyはないため、
送信直前の中断でも手動照合が必要になり得る。別ホスト・別stateで同時に新規作成しない。
復旧時はjournalを保全し、GitHubを照合する。壊れたjournalや欠落した既知の結び付きを空状態へ置き換えない。
通常の読取・status・inspectionはIssueもローカルjournalも作成しない。

履歴は媒体に優先順位を付けず、GitHub comment IDの昇順で結合する。作成/編集時刻では並べ替えない。
同じIDの重複、媒体をまたぐ同じ論理key、同じkeyで異なるpayloadはエラー。
keyは合意/根拠のrevision、レビューのrun＋state/kind、提案のsource、停止のoperation、
noteのgeneration＋body digest等の保存済み識別子から作る。
修正確認・証拠確認は対象PR・指摘/wait・固定snapshotも含む。
識別子のない旧形式では原payload自体をkeyに使い、欠けたfieldを現在値で補わない。
同一媒体内の同一payloadの再掲は、従来どおり履歴として保持する。
片方の取得・検証が失敗したとき、他方の過去の成功記録だけを使って続けない。

liveな共通readerは、選択したstateの `comment-outbox/` と `comment-updates/` に送信・照合済みとして
残る全コメントを、記録種別の許可リストを使わず照合する。親・証拠用Issue・PR投影の保存先、
comment ID、認証publisher、本文全体のdigestが一致しなければ明示エラーにする。
旧version 1のoutboxは親または検証済み証拠用Issueの番号で、新version 2は保存した親番号と媒体でも照合する。
取得済みsnapshotの純粋parserにはこのI/Oを混ぜず、収集時・実行直前のlive読取で検査する。

この完全性検査で検出できるのは、同じstateにjournalが残る記録の欠落・改変である。
別ホスト・別stateが書いた記録、journal導入前の旧コメント、消失・削除されたjournalが示していた過去の存在は
証明できない。旧形式の読取互換は履歴の完全性保証を意味しない。
journalの無い旧親コメントだけで取消が削除されると旧合意へ戻る問題は、origin/main
`2016fa9c4bace66a4d13e98b38841e6565bcbd9c` でも再現した既存挙動である。
journalを削除してエラーを回避せず、remoteと保存済み証拠を運用者が照合する。

証拠側のマーカーなしコメントと別publisherのコメントは、出典URLとpublisher照合結果を持つ
`History.diagnostics` にそれぞれ `unmarked_evidence_comment` / `untrusted_evidence_publisher` として残す。
この診断がある履歴を権限・レビュー等へ投影すると明示エラーになる。
親側の通常コメントは、同じloginの人間の発言も含め、従来どおり指示入力として保持する。
旧合意のcomment baseline・requirements digest等が無い場合も、その欠測を保持する。旧コメントの読取期限は設けない。

### 合意・進捗コメント

コメントの1行目は `<!-- parent-session:v1 <JSON> -->`。
JSONは `parent`、`event` とイベント固有の値を持ち、`<` / `>` をUnicode escapeする。
機械記録の信頼境界は上記readerが扱い、本文中の引用マーカーを無視する。
合意には構造化した `agreement` と、そのcanonical JSONの `agreement_sha256` を含める。
canonical JSONはUTF-8、`ensure_ascii=False, sort_keys=True, separators=(",", ":")`。
再実行時は同一イベントを再投稿しない。別ghアカウントへの移管は自動で信用せず、
新しい担当アカウントで改めて要件確認・合意を記録する。

合意時の本文・コメントのsnapshotは履歴として保存する。変更は `status.input_changes` に表示し、
意味・影響先・既存証拠の適用可能性は親が判断する。合意の本文と変更可能な計画を同一視しない。
実質的な要件変更は、必要な人間の判断を確認して `agree` へ記録する。既存の発言を聞き直さない。
`agreement-ignore-comments` は既存の明示分類を記録する入口として残るが、操作を再開するための必須手順ではない。

`create-child` と `delegate` は、ローカル担当がactiveであり、親がopenであり、
共通履歴に有効な人間合意または自律着手根拠があることを**書き込み前**に検査する。
合意後の分割結果への追加承認は設けない。起票元が人間・レポートのどちらでも登録入口は同じ。
単独で完結する既存issueは登録・合意してそのまま扱い、不要な親子階層を作らない。
分割するときだけ、issues規約に従ってそのissueを親へ移行する。

## 自由文の進捗・連絡

親へ通知する意味判断と4種類の投稿基準は [parent-session skill](../../skills/parent-session/SKILL.md#親issueへ伝えること) が正本。
CLIは意味を推測せず、指定した種類・key・記録・参照と重複を検査する。

`note --body-file tmp/reason.md` は証拠用Issueへの自由文記録。activeな担当本人が要件確認中でも使える。
モデル選択理由・局所の判断・途中結果の保存やdecision_logsのreason_refに使い、親への通知は発生しない。
前後の空白を除き、空・60,000文字超・HTMLコメント記号・予約マーカー名・私的ID等を拒否する。
`event: note`、担当generation、本文digestを保存し、`{parent, body_sha256, url}` を返す。
同じgeneration/本文の再送は同じ記録。noteに要件を書いても要件合意にはならない。

親への通知は `notify --spec tmp/notification.json`。specの共通キーは次のとおり。

| キー | 契約 |
|---|---|
| `schema_version` | 整数1 |
| `key` | 親AIが選ぶ同じ意味の通知単位。英小文字・数字・ハイフン、1〜80文字。世代で変更しない |
| `kind` | `question` / `agreement` / `impact` / `completion` |
| `summary` | 人間が読む要点 |
| `evidence` | 共通readerで検証済みの、この親の先行機械記録のcomment URL配列（1件以上） |
| `resolves` | 解消する保存済みquestionのkey配列。質問自身は解消を指定できない |
| `details` | 下表の種類別object。追加キーや欠落を拒否する |

| kind | detailsのキー |
|---|---|
| `question` | `question`、`resume_condition`（文字列） |
| `agreement` | `change`、`impact`（文字列）、`answers`（1件以上の`{source, quote}`。回答原文と出典を親が固定） |
| `impact` | `result`、`decision`、`next_action`（文字列）、`affected`（対象作業の非空配列）、`achievements`、`remaining`（文字列配列、空可） |
| `completion` | `achievements`、`artifacts`（それぞれ非空の文字列配列。artifactsはhttps URL）、`remaining`（文字列配列、空可） |

まず証拠に `notification_record` を保存し、その実URL/digestへ束ねた `notification` を親へ投稿する。
本文はspecから決定的に生成し、agreementの回答を引用のまま表示する。戻り値は `{parent, key, url, record_url}`。
回答の原文字列は構造化記録に保持する。予約マーカー名を含む正当な引用も受け付け、可読本文ではHTML特殊文字だけをescapeする。
本文中の引用は機械証拠の先頭マーカーとして読まない。これは自由文noteの予約語拒否とは別の契約である。
同じkey/同じspecは元のURLを返し、同じkey/異なる内容はconflictで停止する。
証拠側保存だけで中断しても、同じspecで親通知と進捗同期だけを続ける。新しいAI・実験を起動しない。
`status.notifications` は全種類について `{key, kind, state, record_url, url}` を返す。
`state=pending` は証拠だけが保存された状態で `url=null`、`published` は同じkey・spec digest・
証拠参照・生成本文が一致する親コメントを確認できた状態。statusは通知を投稿しない。
後日の承認や方針変更は新しいkeyと差分を使う。過去の成果や未確認事項を上書きしない。

completionは親ごとに固定key `completion`。手動の全体完了報告もこの入口へ集約する。
`workflow-report completed` は全体の実merge/close/受入を検証したうえで、この完了記録の参照だけを保存する。
完了記録が無ければ、親AIが成果・未達/残件・成果物を明示してnotifyするまで拒否する。
証拠の完了記録があっても親通知が未公開なら `workflow-report completed` と `release --ended` を拒否し、
担当をactiveのまま保つ。同じspecのnotifyで親通知だけを補ってから再実行する。
`release --ended` も公開確認済みの既存完了記録を参照し、全文を親へ再掲しない。担当終了だけで全体完了を生成しない。

コメントの新規POSTは共有stateの `comment-outbox/` でkeyと本文digestを送信前にfsyncする。
再試行で同一publisher/本文が1件なら照合して再利用する。複数・改変・削除はエラー。
送信済みでGitHub上に一致が無ければ `publication uncertain` として再POSTせず、運用者による明示照合まで止める。
送信前中断と応答喪失を「失敗したから再送」で区別しない。共有stateの保全・別ホスト同時作成禁止は上記Issue作成と共通。
唯一の本文更新であるレビューpending→result/adoptionは、同じcomment IDの更新前・更新後digestを
`comment-updates/` にPATCH前に保存し、再取得が一致してからboundとする。
sent中は保存した更新前または更新後の本文だけを許し、bound後は更新後だけを許す。
PATCH応答喪失からの再開でも、確認した確定本文をoutboxへ記録し、以後pendingへ戻った場合は拒否する。
pendingは成功結果として扱わない。旧outboxがpendingを記録したまま更新意図の証拠が無く別本文になっている場合も、
現在値を補って通さず不一致として停止する。

`sync` は進捗概要の `parent-progress:v1` ブロックを保存済み履歴と作業状態から生成する。
段階・未解消の質問とリンク・継続/停止/依存保留・主要成果/残件・最新通知/合意・証拠用Issueへ辿れる。
現在の受入は `completion.status` の適用判定と一致させる。現在の受入が完了し、そのfinishに対応する
workflow reportもある場合だけworkflow完了を表示する。再合意や承認変更で適用されなくなった成果・残件は
「過去」として原通知へリンクする。現在に適用できる受入記録が無ければ現在の受入・残件は未確認と表示し、
記録があればその原文へリンクする。impact通知も後続の合意・取消より前なら過去として扱う。
通知の成果・残件は報告内容として表示し、現在の受入・残件を通知から補完しない。
構造化finishの無い親で条件を編集した場合も同じ判定を使う。
完了記録が無いことを完了や残件0へ補わない。既存の文章とブロック外の人間の追記を保持し、変化が無ければPATCHしない。
取得後に本文が変わった場合は上書きせず停止する。再読後の同じsyncで人間の編集を保持して更新する。
確認・合意・レビュー状態・通知などの更新も同じ同期を使う。担当登録だけでは本文の移行を要求せず、
登録後の現在地表示には `sync` を使う。親ラベルのない単独Issueにも、元の本文・ラベルを保って進捗概要だけを生成する。

機械記録と通知は指示入力・要件digestから除外するが、親側の通常コメントは同じloginでも取り込む。
人間とagentは同じloginを使うため、有効な先頭マーカーを人間が偽装した場合は区別できない。
人間の指示を機械記録へ移して入力から除外してはならない。

### イベントごとの保存と表示

新規親コメントを作るのはnotifyだけ。以下の操作は意味判断を行わず、該当する証拠・本文・PRを更新する。

| 操作/記録 | 保存先 | 進捗概要から辿る内容 |
|---|---|---|
| assigned、checking、handoff_pending/handoff、released/ended | 証拠＋担当registry | 現在の段階・担当状態 |
| agreed、proposal_agreed、execution_basis、basis_revoked、agreement_comments | 証拠 | 最新の合意/原引用・自律根拠、保留の状態 |
| proposal_pending/received、note | 証拠（noteはreason_refの実URL） | 証拠用Issueの詳細、採用/受領の現在地 |
| review targets_fixed/instructed/result_received/fixes_verified/complete | 証拠＋run | レビュー段階・対象・固定証拠 |
| parent-review pending/result/adoption | 証拠。PR側のpr projectionは各PR | 公開済み結果とPRへの参照。pendingは未完了 |
| parent-review resolution/evidence | 証拠＋該当PRのコピー | 解消/証拠待ちと確認先 |
| parent-improvement candidate/published/decision | 証拠（旧親コメントも読取）＋子の参照 | 改善候補・起票・人間判断の詳細への参照 |
| completion（構造化finish） | 証拠＋受入receipt | 各受入条件の確認先。人間向け成果は固定completion通知 |
| issue-start-judgement | 証拠、revision付きで追記 | 判定理由・不足・再開条件。既存判定をPATCHで消さない |
| child_stopped/resumed | 対象本文＋証拠＋registry | 停止/継続/依存保留と再開条件 |
| create-child、delegate、record-result、取消・改善候補 | 正式関係・対象本文・registry、既存判断/workerログ | 作業・成果物・検証・未達/残件 |
| workflow-report completed、release --ended | 完了記録への参照 | 同じ成果/未達/残件。通知の再掲なし |
| notification_record / notification | 証拠 / 親 | 判断待ちと質問、最新通知・成果/残件 |

### 稼働中の親への移行

旧合意・worker・reviewの記録はそのまま読む。次の新規機械記録から証拠用Issueへ保存し、履歴をコピーしない。欠けた本文digestやplanning承認を生成しない。
担当親は現在の合意・本文・コメントと保存済み成果を確認して通常の入口を使う。
旧記録にphaseがなくてもrun/session/終了の証拠が揃えば継続できる。
保存形式の移行が必要な場合だけ [状態の移行](#状態の移行) を使い、現在の実行記録と照合する。

## 自律着手の実行根拠と呼出契約

判定agentの出力を直接登録しない。着手判断側のスクリプトが出典・重要な未決事項・人間の停止/見送り/対象外・
採用状態・正式関係・既存担当/PRを検証し、着手可能な対象だけを次の入口へ渡す。
このCLIは自然言語の判断器ではなく、構造・同一性・鮮度・担当を検査する。

```bash
# 判断材料を固定する。--context は関連コード/依存/採用等のsnapshot参照・digest JSON（省略時{}）
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 basis-input --context tmp/context.json > tmp/inputs.json
# スクリプトが検証済みの判断を構造化してから公開する。担当登録は行わない
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 basis-record --spec tmp/basis-record.json
# 予約直前の読み取り検査。CLIの終了と別プロセスの予約をatomicとは扱わない
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 basis-check --basis-sha256 64桁hex --generation 1
```

記録specは追加キーなしの `{basis, inputs, ignored_comments}`。
`inputs` はbasis-inputの出力そのもの、`ignored_comments` は通常 `[]`。
`basis` は次のversion付きJSON object（追加キーなし）。

```json
{
  "schema_version": 1,
  "kind": "autonomous_decision",
  "purpose": "目的",
  "scope": "対象と対象外",
  "behavior": "期待する動作",
  "acceptance": "完了条件と検証方法",
  "rationale": "この範囲で着手可能と判断した理由",
  "sources": ["https://github.com/owner/repo/issues/1216", "関連コードのcommit/path等の参照"],
  "input_sha256": "inputsのcanonical JSON SHA-256（64桁hex）",
  "policy_version": "autonomous-start/v1"
}
```

文章とsourcesは非空。`human_confirmation` を入れた自律根拠は拒否する。
`basis_sha256` はbasis全体のcanonical JSON digest。人間合意は既存のagreedイベントを
`kind=human_agreement, policy_version=human-agreement/v1` の同形式（実際のhuman_confirmation付き）へ
読み替える。合意原文・agreement digest・公開マーカーは保持し、自律判断からagreedを作らない。
本文digestを持つ人間合意の `input_sha256` は `{agreement_sha256, requirements_sha256}` のcanonical JSON digestとする。
本文digestのない旧合意も元の合意として保持し、現在への適用可能性は親が確認する。
旧review schema version=1のauthorizationは従来どおりagreement digestと照合する。

`inputs` はrepo・親番号・タイトル・要件本文digest・needs-human・指示コメントのID/login/digest・contextを含む。
長い本文・議論を公開マーカーへ複製せず、元の判断入力は収集側が保存する。contextにも小さな参照/digestを渡す。
子タスクのチェックリスト、進捗概要、成果物・関連PR、次の改善候補の節と、Acceptance checklist / 全体の完了条件の完了チェックを除外する。
停止事項・対象外・未知の節は残す。現在のpublisherのparent-session:v1 / parent-review:v1記録は
進捗として分離する。本文・指示の変化は親が確認するための入力であり、既存作業を一律失効させない。
判定コメントも人間合意と同じ厳密な形式検証で分離し、新しい根拠の `ignored_comments` には固定しない。
旧根拠の除外リストに判定表示がある場合は、現在のコメント全体を検証できたものだけ表示のdigest変更を許す。
それ以外の明示除外は従来どおり本文変更・削除を拒否する。

無関係な通常コメントは、内容を確認した呼出元だけが
`--ignored-comments` に `[{"id":123,"sha256":"comment_snapshotのdigest","reason":"要件と無関係な理由"}]`
を指定できる。`parent.execution.comment_snapshot` は `{id,login,body}` を返す。
同じ分類を記録specへ渡す。除外対象の本文変更・削除・重複IDは拒否する。
分類だけを更新してinputsとbasisが同一ならbasis digestは変わらず、既存のレビュー確認・許可を失効させない。
人間の指示を無関係として除外してはならない。新しい自然言語の意味の確認を、digest照合で代用しない。

後続スクリプトのPython API（import rootは `.agents/ops`）:

- 判断側: `parent.execution.execution_input(session, ignored_comments=(), context=None)` と
  `record_autonomous(session, spec)`。戻り値は `{parent,basis_sha256,generation}`。
  未担当なら環境session IDは不要。activeな担当があれば担当本人だけが更新でき、引継ぎ待ちは拒否する。
- dispatcher: **同じ `Store.locked()` 内で** `validate_candidate(session, basis_sha256, generation)` を呼び、
  返るauthorizationを保持して既存registryへ予約する。対象open・最新入力・最新根拠・担当不在・次generationを検査する。
  予約・実起動・runtime identityとの結合と復旧はdispatcherが実装する。別の担当台帳を作らない。
- 親workflow: 担当登録後に `session.agreed()`（既存名。有効な両種の根拠を受理）と
  `parent.execution.authorization(session)` で再検査する。委任・レビュー前・merge時にも再照合する。
  contextの外部資料・依存・採用状態の再取得と意味の検証は呼出元の責任で、contextの自己申告だけを鮮度証拠にしない。

自律根拠は記録時の次/現generationへ結び付く。担当交代・再登録後は明示的な再判断を要し、旧委任を再起動しない。
根拠・担当が変わっても、保存済みreviewの対象と実行記録は保持する。
`basis-revoke --reason-file` は親全体の明示的な取消。`checking` は確認中の表示であり取消ではない。
一部だけの停止は子の `stop` を使う。新しい根拠が必要か、どの証拠を再利用できるかは親が判断する。

## 採用提案の受領

日次triageは指定親のactiveな登録を確認し、証拠用Issueに `event: proposal_pending` を記録する。
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
GitHubコメントから復元する。明示的な `basis-revoke` は親全体と提案別の合意を取り消すため、
親の `agree` と各提案の合意を再記録するまで、それぞれの実装を拒否する。
既存提案の合意内容を変更する場合も `basis-revoke` で明示的に取り消してから記録する。
未取消の合意は同じ内容の再送だけを許可し、レビュー後に対象作業の要件だけを黙って差し替えない。
親未指定・未登録・停止中なら受領できない。人間の指定/再開後に親自身がregister/handoffして同じ入口へ戻る。
採用チェック・選択肢・提案本文が後から変われば起票/委任時も拒否し、担当親が採用と要件を再整理する。
受領・起票・委任時は、出典がopenで完了済みの `ops-report` issueであることも検査する。
triageの引渡し・受領・提案別合意は既存のマーカー付きイベントとして記録する。
担当親の記録と通知は [専用入口](#自由文の進捗連絡) を使い、既存合意のコメント鮮度を保つ。
人間が独立に作ったissueには受領操作は不要で、同じregister/checking/agreeから開始する。

## Issue自身を実行する

Issueの分割基準と実装者の選択は [parent-session skill](../../skills/parent-session/SKILL.md#親が判断すること)を参照する。
一つのバグ修正を実装・テスト・README更新までまとめて受け入れる場合は、同じIssueのまま進める。
独立してmergeできるAPIとその利用側を別担当で並行に進める場合や、一方だけを保留する必要がある場合は分割が役立つ。
一Issueを常に一PRへ制限する契約ではない。

`work_items()` が全入口の対象を解決する。正式な直属子があればopen/closedともその子だけ、
正式な子がなく上位親もなければ `issue:parent` の有無によらず自身一件を返す。
入れ子の手動親は従来どおり自分の子を管理する。正式な上位親を持つ葉Issueをこの入口で切り離して実行しない。
取得失敗はエラーのまま扱い、子なしや全子完了への変換はしない。
CLIの `--child N` は担当Issue番号の指定で、自身の番号も受け付ける。子作成や自己参照のsub-issue登録は行わない。

担当登録・有効な実行根拠を揃え、専用worktreeで実装して `Closes #1216` を持つPRを作る最短例:

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 register
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 agree --spec tmp/agreement.json
# 親が直接実装した後、差分・受入条件・対象HEADのテストを確認する
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 record-result --child 1216 --evidence-file tmp/tests.md
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 review-request \
  --prs 1250 --instruction-file tmp/review.md --evidence-file tmp/tests.md > tmp/review-request.json
```

直接実装にはdelegate・架空のworker・終了記録は不要。
同じ成果をworkerへまとめて依頼する場合だけ、実装前に以下を使う（worktreeは作成済みとする）。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 delegate --child 1216 \
  --agent codex --branch codex/small-fix --worktree /path/to/main/.claude/worktrees/small-fix
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 run-worker --child 1216 --prompt-file tmp/task.md
```

以後は両経路とも同じ [正式レビュー要求](#レビュー要求契約schema-version-3version-1と2の読取互換) と
[merge gate](../review/README.md#merge-gate)、全体受入・[終了照合](development_workflow.md#待機完了担当終了)を使う。
レビュー指摘や必要なmerge後確認は同じworkerへの継続で扱い、確認専用Issueや必須planningを追加しない。
`sync` は自身の状態を返し、進捗概要だけを更新する。親ラベルや自身のチェック行、子用要件を作らない。
空の子一覧・PRのclose・runner終了だけでは完了せず、成果受理、正式レビュー、対象HEADのCI、実merge/closeと実行終了を照合する。
merge後の確認をworkerへ継続した場合は、そのターンの実結果を親が確認し、`record-result` で受理してから全体完了へ進む。
その既存workerの停止・再開も同じIssueで扱い、Issueの再openや新しい委任先は作らない。
worker来歴と親判断は既存のログ形式へ親番号＝担当Issue番号として接続し、ログ専用の作業Issueは要求しない。

途中分割では実行・PR・受入済み成果を自動で別Issueへ移さない。自身への予約や成果が残る間の `create-child`、
分割で自身の作業が対象外になった場合、未終了workerが正式対象から外れた場合、全子を切離した後の自身への切替は理由付きで拒否する。
終了を照合して別の正式親へ移した子の履歴は保持し、残った独立子は従来どおり進められる。
移管済み提案の作成を旧親から再送した場合も、実番号と正式親を返して再作成しない。旧親の対象整理が必要なら `target_conflict` を明示し、起動許可にはしない。
既存の履歴は保持する。成果とPRのない自身の作業は、明示的な停止とworker終了の照合後に分割できる。
分割と稼働が重なった場合も既存の自身の作業への `stop` / `stop-worker` は使えるが、継続は整理が済むまで拒否する。
成果やPRがある場合は、担当親が元の関係とPRの帰属を明示的に整理してから進める。記録を消して新しいworkerを起動しない。

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
`PR #番号` と `/pull/番号` を指すMarkdownリンクは参照PRとして区別し、依存issueには数えない。
明示した場合はその値で依存欄も生成する。依存先は同じ親の既存の直属子に限り、循環は拒否する。
既存の自由文に後続タスク番号が混じる場合は、`delegate --dependencies 番号...` で先行だけを明示する。
進捗判定は常に最新の子本文の依存欄を読む。明示CLI入力は委任時に本文へも反映し、古いローカル記録で上書きしない。

生成本文の末尾に `<!-- parent-task:v1 parent=N key=KEY spec=SHA256 -->` を付ける。
open/closed全issueをページングしてキーを照合し、spec不一致・複数一致を拒否する。
既存があれば再利用し、ラベル・正式なsub-issue・親一覧の不足だけを揃える。
作成応答の喪失や関係登録・親編集の途中失敗も、**同じキーとspec**で再実行する。
別の正式な親へ所属したissueを自動付替えしない。stateの消失だけを理由に新しいキーで作り直さない。
手動のレポート由来specは受領・その提案を含む要件合意・現在の採用を検証し、共通 `index_proposals` でも照合して、既存提案を別キーで二重起票しない。
triageの `ops-report-task:v1` 出典を持つ既存作業は [自動Issue化](../ops-report/README.md#採用提案の自動issue化) を参照する。
その子の委任では、有効な独立の `autonomous_decision`、現在の正式親、open/完了レポートの採用・選択肢・内容digestを
`adopted_tasks.py` で照合する。関連親の手動記入や架空のproposal_received/proposal_agreedは要求・生成しない。
これは出典の検証経路だけを追加し、担当・generation・レビュー・mergeの条件を変えない。
開発workflowは `adopted_tasks.check_task_adoptions(gh, body)` を親と対象子の両方へ実行・merge直前に呼ぶ。
単一anchorを持たない共通親・分割した子にもcontext出典があるため、`parse_proposal_source()` だけで省略しない。
返却は出典の確認であり実行根拠ではない。既存のauthorizationとdispatcherの鮮度検査も必要。
同じ提案の既存子はキー・タイトル変更・親付替え後も再利用し、現在の正式な親を返す。自動で親を戻さない。
親関係のない既存提案が別キーなら番号を示して拒否し、孤立issueを明示的に移行・照合してから扱う。

`delegate` は実在する専用git worktree・branch・agent prefixを確認し、起動予約を先に保存する。
未分割のIssueは親ラベル付きでも自身の番号を `--child` に指定できる（[対象解決](#issue自身を実行する)）。
`action: launch` を受けた手動親は [run-worker](#手動workerの起動継続とrun記録) を呼ぶ。`delegate` 自体は起動しない。
既存PR・委任予約・完了・取消・停止・依存保留・予約上限・利用できる依存PRの不在があれば起動を拒否する。
`reconcile` は正式な子、全状態のPRの `Closes/Fixes/Resolves #子番号`（同repoのURLも可）、
記録済みbranchを照合して、PRと担当・検証記録を返す。
PRは子番号ごとのclosing指示を明記し、複数子を閉じる場合も各番号に指示を付ける。
ブランチ名だけから親子関係を推測しない。
既存PRや予約がある場合の `do_not_launch` は再開の調査対象であり、成功や完了を意味しない。
予約後の起動失敗・中断時は、既存workerが終了した証拠を確認し、必要なら停止・再開を記録して
`delegate --restart-evidence-file tmp/evidence.md` で明示的に再予約する。既存PRがあれば再予約もしない。

`record-result --child N --evidence-file tmp/tests.md` はclosing指示とbranchを照合し、
担当Issueの成果物欄へPRと検証結果を追記する。記録だけで完了扱いにせず、過去の検証記録も残す。
委任予約にはauthorizationと子の要件本文digest、結果記録にも現在のauthorizationと要件digestを持たせる。
根拠・担当・子の要件が変わったときは親が成果への影響を判断し、必要な部分を再確認してrecord-resultへ記録する。
要件確認中の結果記録は従来どおり可能だが、authorization=nullの事実記録であり実行の許可には使わない。
自身を対象とする結果も同じ入口を使い、子一覧は生成しない。

### 並列起動の選択と予約上限

`status` / `reconcile` / `sync` / 成功した `delegate` は `launch_plan` を返す。
`launch.py` が最新の子本文・正式関係・全状態のPR・既存予約を使い、次のJSONを計算する。

```json
{
  "policy": {"schema_version": 1, "max_parallel_children": null, "reason": "設定の根拠"},
  "active_reservations": [1217],
  "available_slots": null,
  "launchable_children": [1220],
  "tasks": [{
    "child": 1220, "state": "continue", "action": "ready", "dependencies": [1217],
    "stacked_on": [{"child": 1217, "pr": 1250, "branch": "codex/upstream", "head_sha": "40桁SHA"}],
    "reasons": []
  }]
}
```

`tasks` は全直属子を番号順で含み、`reasons` は `{code,message}` の配列。
起動対象は `action=ready`、対象外は `do_not_launch` と具体的理由を持つ。
既存PR・予約・完了/取消・停止・停止先への依存・依存PRの不在/複数/別repo・親のphase/担当・上限待ちを区別する。
未完了の推移的な依存にも同repoのPRが一意に必要で、閉じた未merge PRは利用しない。
停止した依存先のPRがあっても起動しない。完了済み依存はstackの対象から外す。
計画は予約・起動許可そのものではない。実行根拠の鮮度、提案の採用、専用worktree、依存headは
`delegate` が書き込み直前に検査する。`status` の読み取りだけでは予約もagent起動も行わない。
`sync` は親本文の進捗概要に予約一覧・起動可能・待機理由を反映する。

手動親の既定は `max_parallel_children=null`（依存が許す全子）。上限があれば空き数まで番号順に選ぶ。
上限と根拠は親のregistry行の `launch_policy` に保存し、同じ `Store.locked()` で委任予約と直列化する。
設定はworktree・セッションの再開をまたいで維持する。未知schema・不正型は既定値に置換しない。

```bash
# 実測で上限が必要なら、根拠を非空のファイルに記録して設定する
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 configure-delegation --max-parallel-children 4 --reason-file tmp/resources.md
# 制限を解除する場合も理由を残す
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 configure-delegation --max-parallel-children all --reason-file tmp/resources.md
# runnerの終了・子孫の不在を確認した親が、delegateで返ったdelegation_idを指定する
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 record-worker-ended --child 1217 --delegation-id 32桁hex --evidence-file tmp/worker-ended.md
```

上の4件は設定方法の例で、推奨初期値ではない。固定3件の根拠だった事故は、ホストRAM上限24GB・GPU 16GBの環境で
2026-10-18にpytestとtraining jobが重なってPCが落ちた事例。子の数の実測上限ではないため、
pytestは他の親・子と並列実行できる。GPU・Blenderはtraining queueの既存 `half` / `all` 枠で調停し、子の上限は追加の実測に基づき設定する。
自律ループの「各親の実装子2件」という初期値と実プロセスの稼働枠・担当排他はdispatcherの責務で、ここでは変更しない。
dispatcher/親workflowはその実行枠内でこの計画を使い、必要なら同じ予約上限を明示設定する。

終了未確認の委任予約は停止・PR作成・子のclose・親の待機だけで解放しない。
`record-worker-ended` は親が確認した終了証拠を現在の `delegation_id` へ結び付けるだけで、プロセスを停止しない。
旧予約でIDがないものには明示値 `legacy` を使う。再委任は新しいIDになり、古い終了通知は拒否する。
予約枠を解放しても `delegated` は保持し、自動再起動はしない。新しい委任への再予約は従来の証拠付き `delegate --restart-evidence-file` を使う。
停止・根拠・PR検査は再予約でも省略しない。稼働中workerの終了を推測する新しい台帳は作らない。

### 手動workerの起動・継続とrun記録

activeな担当本人が `delegate` の予約を確認して次を呼ぶ。共通引数のstateは予約時と同じものを使う。
会話ではこの前景コマンドをバックグラウンド実行し、独立した子を続けて起動できる。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 \
  run-worker --child 1217 --prompt-file tmp/task.md
# 高難度の委任で親が代替モデルを選ぶ場合。理由は証拠用Issueのnoteにも残す
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 \
  run-worker --child 1217 --prompt-file tmp/task.md --model gpt-6-astra --model-reason-file tmp/model-reason.md
# 同じ委任・sessionの修正ターン。終了枠を解放済みなら再開証拠も渡す
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 \
  run-worker --child 1217 --resume --prompt-file tmp/fix.md --restart-evidence-file tmp/restart.md
# この入口の導入前に共通ランナーで起動したrunを、稼働中・終了済みとも登録できる
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 \
  record-worker-run --child 1217 --run-dir "$existing_run"
```

`run-worker` は現在の32桁hexの委任ID・Codex担当・worktree・branchと、有効な根拠・要件・停止・依存を検査する。
初回は終了未記録かつ当該委任のrun未登録に限る。`--model` の選択肢は
[`agents.conf`](../lib/agents.conf) の `CODEX_MODEL`（既定Sol）と `CODEX_ALTERNATE_MODEL`（Astra）のみ。
effortはmax固定で引数に出さない。判断基準は [AGENTS.md](../../../AGENTS.md#開発ルール) を参照する。
高難度の委任では親がAstraを明示選択でき、人間との合意による指定を優先する。
選択理由は `note`（証拠用Issue）と `--model-reason-file` に残す。初回の代替選択には理由ファイルを必須とし、
既定の採用で省略した場合は既定採用の理由を記録する。委任taskの `model` / privateな `model_reason` と各runへ保存する。
共通ランナーを `--role implementation --mode write --task-id 委任ID --cwd worktree --branch branch --model 選択値`
で呼ぶ。選択値・実効値・設定の出典はrunnerの `contract.settings` と `runtime.effective` で照合する。
利用不可・実効値不一致はエラーとし、別モデルへのfallbackは行わない。
Codex CLIを直接起動せず、この入口から共通ランナーへ渡す。
指示は `start-worker` / `continue-worker` と同じ `worker_instruction.py` で組み立てる。
[標準のworker指示](worker_prompt.md)、親・担当issueの番号、worktree・branch、担当issueの現在の原文の後に、
`--prompt-file` の内容を「親の具体的な指示」として続け、共通ランナーへ標準入力（`--prompt-file -`）で渡す。
継続のターンも同じ構成で、指示ファイルには案件固有の内容だけを書く。

runごとに `state dir/worker-runs/委任ID/UUID/` を割り当て、共通ランナーの起動前にregistryへ保存する。
実行中はregistry・親の操作ロックを保持しない。終了値は共通ランナーの値を返し、stdoutは
`{child, delegation_id, exit_code, runner_summary}`。`runner_summary` はランナーの公開要約JSONであり、
起動前拒否等で要約が無い非ゼロ終了ではnullとする。最終文のprivateな参照先は `final_file` へ返す。
起動前の明確な拒否は診断を残して予約を復元し、起動有無が不明な中断は予約とrunを保持する。別sessionを自動起動しない。

`--resume` は現在の委任で最後に記録したrunの契約と終了・cleanupを検査し、
共通ランナーの `resume --resume-from` で同じsessionへ戻す。委任IDは共通ランナーのtask IDなので保持する。
model省略時は直前runの保存contractから継承し、同値の再指定は許可する。異値はrunner起動・run追記・予約枠変更より前に拒否する。
導入前のmodel欄がない委任も保存contractの選択値から復元し、判別できなければ拒否する。既定値とは推定しない。
以前のrunにmodel overrideがなければrunnerへの引数も省略し、`overrides` / `selection_sources` の契約を保持する。
理由は省略時に引き継ぐ。導入前の理由未取得はnullのまま残し、`--model-reason-file` で親の理由を補足できる。
`record-worker-ended` 済みなら保存済みrunの終了・回収と空き枠を照合し、
同じトランザクションで終了記録を取り除き、予約枠を取り直して継続runを保存する。
終了未記録なら予約枠を保持したまま継続する。既存PRがある修正も同じbranch・worktreeを使う。
`launch_plan` の停止・依存・上限判定は省略しない。再開証拠はprivateな `restart_evidence` に残す。

`record-worker-run` は存在する `result.json` / `invocation.json` のすべてでschemaと
`contract.task_id`・`contract.cwd`・`contract.branch` と実装子のmodel/max effort・profile・modeを現在の委任と照合する。
複数の保存記録間でmodel設定が異なる場合や、modelを特定できない場合も拒否する。
どちらも無い場合や片方でも不一致なら拒否し、同一runの再登録は冪等とする。
起動はせず、既存の終了記録は変更しない。保存contractから委任のmodel欄を補完する。

`delegate` は委任予約を保存するとき、子taskのprivateな `worker_runs` を必ず設定する。
既存の記録は保持し、なければ空配列にする。このキーがある子は `run-worker` / `record-worker-run`
でrunを記録する方式の委任であり、現在の委任IDに対応するrunがなければ「委任予約済み・未起動」を意味する。
`delegate --restart-evidence-file` による再委任でも旧委任のrunを保持し、新しい委任IDのrunがない状態から始める。
新しい委任のtaskのmodel・理由はクリアし、初回に選び直す。旧runのmodel・理由はそのまま保持する。
この変更より前の、キーを持たない旧委任は「run記録なし」であり、未起動とは区別する。
読取だけで旧委任へキーを補わない。

`worker_runs` は次の6キーに新しいターンの `preflight` snapshotを加え、記録順で保持する。導入前の4/6キー記録も読み取る。

```json
[{"delegation_id":"32桁hex","run_dir":"worker-runs/委任ID/UUID","kind":"initial","recorded_at":"2026-10-30T00:00:00+00:00","model":"gpt-6.1-sol","model_reason":"通常の実装として既定を選択。"}]
```

`kind` は `initial` / `resume` / `registered`。`run_dir` は親state dir基準の相対パスで、
state外の登録runは絶対パス。`recorded_at` はUTC ISO8601。
version 1/2とも既存フィールドと過去の委任のrunを保持し、新しいrunだけ追記する。
`status`・`reconcile`・委任応答からこのフィールドを除外し、issue本文・コメントにもrun dirを出さない。
再開証拠等の文字列に同じrun dirが含まれる場合も、保存原文を保持してCLIでは `[private run]` へ伏せる。
実行状況の読み取り側はこのprivate registryを参照する。CLI出力を別のrun台帳として使わない。

子taskの `title` は実行状況に表示するタスク名で、`create-child` と `delegate` が記録する。
親・子issueは題名の書式から取り出した要約、それ以外のissueは題名全体を保存する。
`sync` は既存の子taskの名前をGitHubの現在の題名へ更新する。ローカルtaskのない正式な子へは追加しない。

### workerの調査・親判断・実装フェーズ

各ターンの作業は親の具体指示で決める。既定は `implementing`、調査だけを依頼するときは
`run-worker` / `start-worker` / `continue-worker` の `--planning` を使う。
planningの報告は任意の書式でよく、親は直接 `--resume --prompt-file` で次の作業を指示できる。
回答を報告と結び付けて保存したい場合は `decide-worker-plan --action approve|investigate|reject --answer-file` を使う。
保存済み回答は次の同session継続に送られ、具体指示や追加指示も併記できる。

`preflight` はターン・指示・回答・runtimeの来歴。本文・担当世代・依存の状態変化で自動失効させない。
新しいtask記録はversion=2。旧version=1も読み、次の明示操作で新しい来歴を追記する。run/jobのsnapshotはversion=1のまま保持する。
旧CLIは新しいtask versionを扱えないため、呼出元を新CLIへ更新してから利用する。稼働中の記録を一括変換しない。
statusの `inputs_changed` は親が意味と影響を確認するための通知であり、再調査の強制ではない。
旧委任にphase記録がなくても、通常のrun・session・終了照合を経て継続できる。過去の承認を生成しない。
過去のrunとレビューは作成時の対象を保持し、今の条件への適用可能性は親が判断する。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 \
  run-worker --child 1217 --prompt-file tmp/task.md
# 調査を先に依頼する場合だけ --planning を追加
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 \
  run-worker --child 1217 --resume --prompt-file tmp/next.md
```

通常再開の終了・cleanup・同一sessionは保存済みrunner記録から照合する。`--restart-evidence-file` は補足任意。
PRがmerge済み、issueがclosed/completedでも、同じworkerへ残る受入確認を依頼できる。
取消・明示停止・生存不明・worktree/branchの取り違えは別に検査する。
`run-worker` は最新runの `final_file` を返す。起動前拒否でrun dirが作られていない場合は
失敗の診断をprivateな履歴へ保存し、実行runと予約を元に戻す。起動有無が不明な場合は予約を維持する。

ログ保存と公開は [workerログ保存](../worker_logs/README.md) の独立した結果。
`pending` は `publish-worker-log` で再開し、workerの終了や他の作業の可否へ読み替えない。

### stacked branchの確認とrebase

`stacked_on` の各PRはclosing指示で依存子に対応付け、branch名だけから推測しない。
親が全headのコード・インターフェースを確認し、同じrepoに取得してから専用worktreeを作る。
以下は依存先が1つの場合。`upstream_head` は計画の完全なhead SHAで固定し、ローカルに保存する。

```bash
git fetch origin codex/upstream
upstream_head="$(git rev-parse origin/codex/upstream)"
# 計画のheadと一致しなければ再照合する
git worktree add -b codex/downstream .claude/worktrees/downstream "$upstream_head"
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 delegate --child 1220 --agent codex --branch codex/downstream --worktree .claude/worktrees/downstream
```

作成先は必ずメインcheckoutの `.claude/worktrees/`。親自身がworktree内なら上の相対パスをその絶対パスへ置き換える。
複数依存なら全headを含むbranchへ積むか、子専用の統合branchへ取り込む。
`delegate` は全未完了依存headが子のHEADの祖先であることと、保存直前にもhead/依存が同じことを確認する。
予約の `stacked_on` が起動時の境界。先行PRの更新を取り込んだ場合は、実際に積み直した境界もローカルへ記録する。
後続PRのbaseは先行branchにして子だけの差分を表示し、merge順序は依存順を維持する。

先行PRのmerge後は、子worktreeがcleanであることを確認し、保存した境界から子自身のcommitだけをmainへ載せ替える。
通常merge・squash mergeのどちらでも、子の差分から先行変更を除く。

```bash
git fetch origin main
git -C .claude/worktrees/downstream rebase --onto origin/main "$upstream_head" codex/downstream
git -C .claude/worktrees/downstream diff origin/main...HEAD
```

全先行PRのmergeを確認してから子のPR baseをmainへ変更し、差分・テスト・依存を再確認する。
独立した複数依存の統合branchでは、全先行headを取り込んだ統合commitを境界に保存して同じ `--onto` を使う。
競合は親が解消・検証し、判断が必要ならその子を停止する。head/base/diff変更は既存review coverageで承認済みにせず、
rebase後にレビュー対象を固定する。既にレビューした対象の変更には理由付きの新しい明示要求が必要で、自動再レビューしない。

### pytestの並列実行

pytestのホスト全体の共有lock制約は廃止した。親・子・worktree間で並列実行してよい。
GPU・Blenderのtraining queueは従来どおり使用する。

```bash
bash .agents/ops/pytest_locked.sh tests/e2e/agents_ops/test_parent_launch.py -q
# worker数も実行内容に合わせて指定できる
bash .agents/ops/pytest_locked.sh tests/e2e/agents_ops/test_parent_launch.py -n 2 -q
```

既存の呼出元のためファイル名は維持するが、`pytest_locked.sh` はlockを取得しない。
自身のworktree rootへ移動し、その `.venv/bin/python -m pytest` を実行する。
worker数は指定がなければ `-n 4`、指定があればその値を使い、pytestの終了値を返す。
`PYTEST_LOCK` と `XDG_STATE_HOME` は実行制御に使わない。
`.venv/bin/python -m pytest` の直接実行も許可する。

旧版のworktreeや起動済みの `flock` はmainの更新だけでは変わらない。
待機中の検証はその起動元で中止してから、変更を取り込むか、同じworktreeで
lockなしのpytestコマンドを起動する。lockファイルの削除による解除は行わない。
テストでは2つのworktreeの同時実行、旧lockの保持中でも開始できること、
worker数の指定と失敗終了値の保持を検査する。

## 停止・再開・改善候補

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 stop --child 1217 --spec tmp/stop.json
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 resume --child 1217 --spec tmp/resume.json
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 improvement --child 1217 --key result-access --spec tmp/improvement.json
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
Project表示は明示的な `project-setup` / `project-sync` を使う。既存コマンドからは呼び出さない。
項目・ビュー・節目spec・担当・読み取り専用PMとの境界は [Project表示](../project_progress/README.md) を参照する。
`closed/completed` だけを完了チェックにし、取消は未チェックで残す。
親全体の完了はチェック数から推測せず、全体の受入条件を親agentが確認する。

改善候補の正本は元親の構造化コメント。`--child` は発見元で、省略時は親自身を表す。
子の欄には候補への参照を残す。既存の `purpose` / `evidence` / `scope` だけのspecは未整理の発見記録として受け付ける。
新親の起票には同じkeyで境界を補い、`improvement-publish` を明示する。人間の判断は `improvement-decide` で記録する。
候補・起票・判断・実行は `status.improvements`、今回の受入は `status.completion` に分けて表示する。
schema・冪等な再送・承認の保証範囲・導入順は [終了時改善と全体受入](improvements.md) を正本とする。

## 担当の終了・引継ぎ

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 handoff --note-file tmp/handoff.md
# 指定された次の親セッション自身が実行する
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 handoff --accept --note-file tmp/takeover.md
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 release --ended --spec tmp/finish.json
```

`handoff` で引継ぎ待ち、`release` で解除、`release --ended` で終了済み。
構造化finishは[合意の受入snapshot](improvements.md#今回の受入と候補整理)を照合する。
従来の `release --ended --note-file` も使えるが、全体受入を検証する操作ではなく「受入未確認の担当終了」となる。
引継ぎ待ちは旧担当も分割・委任できず、新担当のacceptまで代替起動しない。
acceptは通常、解除/終了/引継ぎ待ちだけを受け取り、同時担当数を再検査してissue/PRを照合する。
中断した同一セッションは `reconcile`、別セッションは明示acceptから始める。
旧セッションが未解除のまま失われた場合は、ローカルstateと既存workerを確認し、
人間の担当指定に基づき `handoff --accept --recover --note-file tmp/recovery.md` を明示実行する。
ノートに旧workerの停止・既存worktree/PRを確認した証拠を残す。通常のregisterはactive担当を上書きしない。
recoverも既存の委任予約を保持し、照合結果から自動再起動しない。stateを消して空き担当と扱わない。
引継ぎノートには完了/未完了の子・PR、停止条件、次の作業、レビューrunを記載し、ローカルIDを書かない。

終了・引継ぎを確定した後、CLIはworkerログの残件、今回の終了結果の捕捉、未公開の判断ログを順に照合する。
公開の失敗は担当変更を取り消さず、`publication_exit` に残件・担当・再開コマンドを返す。
自律親も既存supervisorの終了イベントから同じ照合を行う。通常の書込み操作は共通後処理から、
workerの起動・継続は予約前に一度だけ保留公開を続ける。status系とdry-runはstateを書かない。
対象入口、他親／journal不在PRの明示回収、読取りAPI `remaining_publications` と
privateな終了checkpointの型・保存先は [workerログ契約](../worker_logs/README.md#続行一括回収終了checkpoint) を正本とする。

## レビュー証拠の照合表示

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 review-inspect --prs 1250 1251
```

prepare前に、既存の証拠・不足・現在との差をPRごとにJSONで読む。
requestやstate・GitHubを書き換えず、agentを起動しない。担当セッションIDなしでも使える。
出力・不明範囲・公開とcoverageの検査・終了値の正本は
[reviewの照合表示契約](../review/README.md#既存証拠の照合表示) を参照する。
意味の判断と次の行動は親が記録し、追加レビューが必要なPRだけを以下の入口へ渡す。

## レビュー要求契約（schema version 3、version 1と2の読取互換）

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 review-request \
  --prs 1250 1251 --instruction-file tmp/instruction.md \
  --evidence-file tmp/result.md > tmp/review-request.json
```

`review-request --prs` は成果の登録・対象固定・明示指示を1コマンドで行う。
子の終了は共通ランナーの記録から検査する。自律workerはdispatcherの現在のjob、
手動workerはregistryの `worker_runs` から現在の委任の登録runをすべて照合する。
手動の終了記録があっても契約と終了を再照合し、記録なし・不一致・未終了なら拒否する。
未委任の子はrunの検査対象外。runの導入前のworkerは `record-worker-run` で紐付けてから要求する。
終了・cleanup・worktree・branch・task IDの一致なしに予約を解放しない。
従来の分離操作 `record-result` → `review-prepare` → `review-request --run-id` も使える。

対象は直属子にclosing指示があるopen PRの任意の部分集合。
PR未取得・重複・closed・draft・fork・親と無関係をエラーにする。
他のopen PRの追加・更新・closeは選択した対象のcoverageを失効させない。
分割していないIssueでは親ラベル付きでもそのissueを閉じるPRを同じ方法で扱う。
GitHub取得・git差分取得後にPR集合とhead/baseを再取得し、途中更新を拒否する。
不足オブジェクトだけ `origin` からSHAをfetchする。

prepareはversion 3を作成する。従来のversion 2のJSONは次の形で、読取時に書き換えない:

```json
{
  "schema_version": 2,
  "authorization": {
    "generation": 1,
    "execution_basis_kind": "autonomous_decision",
    "execution_basis_sha256": "basis全体の64桁hex SHA-256"
  },
  "requirements": {"1216": "親の要件本文の64桁hex digest", "1217": "対象子の要件本文の64桁hex digest"},
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

version 3は上のversion 2へ`review_selection`を追加してdigestに含める。
選択JSONの定義は[共通ランナー](../lib/README.md#レビュー専用の選択入力)を参照。
既定fixedでは引数を省略すると設定の固定値を使用する。明示指定は固定値と一致する場合だけ許可する。
selectでは`--review-model`と`--model-reason-file`を必須とし、必要なら`--review-effort`も指定する。
通常確認・高リスク確認などの意味判断は親が行う。CLIは行数やパスからモデルを決めない。

これらの引数は`review-prepare`と`review-request --prs`の両方で使える。
`review-request --run-id`は保存選択を使用し、指定を加えた場合も完全一致を要求する。
理由・モデル・選択元の変更は、新しい理由付きprepareで別runにする。prepare済みmanifestを変更しない。
設定の正本は[lib/agents.conf](../lib/agents.conf)で、明示設定の検証には`--review-config-file FILE`を使える。
実行側の`parent_review.py run`にも同じ内容の設定ファイルを指定する。

人間の指定を親が確認した場合は`--review-model-source human_instruction`と`--human-model-reference`を使う。
別の親候補との矛盾も検査する場合は`--human-review-model`、必要なら`--human-review-effort`で拘束を渡す。
人間の発言であることと元指示の意味は親が確認し、CLIは拘束との矛盾・許可外の組を拒否する。
人間指定がfixedと矛盾しても固定モデルへの振替は行わない。

保存finalの復旧を指示するときは、prepare時に`--recover-source-run ID`で元runを固定する。
この選択は`{kind:recover,source_run,source_request_sha256,candidates}`となり、新しいモデル選択とは併用しない。
`candidates`には元requestが継承していた未採用候補を保持し、recoverを経由して指摘・証拠待ちが消えることも防ぐ。
復旧要求を`run`へ渡すことも拒否する。実効モデルはrecoverが検証する元runの値である。
別モデルの追加レビューには`--handoff-from-run ID --handoff-scope-file FILE`を使う。
元request/resultのdigestとPRの対応を固定し、元run・未解決事項・追加確認範囲をcontextへ渡す。
検証済みの未採用候補があれば、その指摘・証拠待ちと検証来歴もrequestへ固定する。
連続する引継ぎでも候補を保持し、不正な候補や固定後の改変は明示エラーにする。
追加レビューの理由は従来の`--reason-file`、モデルを選ぶ理由は`--model-reason-file`に分ける。
引継ぎはcoverageの合成でも旧指摘の解消でもない。部分確認だけならnot_reviewedのままにする。
候補が残るPRのgate解除には、[保存候補の引継ぎ契約](../review/README.md#保存候補を含む別runへの引継ぎ)に従う親の明示確認も必要になる。

旧version 1/2には要求時の選択を補作しない。既存証拠の復旧・公開再開・gateは旧契約として検証する。
未起動の旧requestは従来のfixed設定でだけ起動でき、selectや固定値変更時は新版requestを要求する。
手動親と自律親は同じ入口・検証を使う。選択元は判断記録の既存の3分類へ接続する。

証拠用Issueの `event:review, state:targets_fixed` にrun ID・requestのcanonical JSON SHA-256を記録し、
本文のJSONコードブロックにmanifest全体を載せる。この時点ではレビュー未指示。
`review-request` は同じ対象が現在も有効なことを再検査して、親の明示指示を証拠用Issueへ記録する。
出力は上記manifestに `"review_state":"instructed"` を追加したJSON。
reviewイベントには操作時のauthorizationを記録し、manifestは作成時の来歴を保持する。
requirementsは親・対象子の本文の観測記録。文字列変化による自動失効には使わない。
prepare中の再取得・指示・収集時はPRとclosing対象の同一性を照合する。
実際のコード・CI・指摘解消はreview/mergeで照合し、要件の意味と変更の影響は親が判断する。
保存済みmanifestと過去のrunを新しい承認へ書き換えない。
`request_sha256` は `review_state` 追加前のprepare manifestのcanonical JSONに対して計算する。
**#1218はこのJSONだけで起動せず、共通readerの同じrun ID/request digest/指示済み記録と最新対象を照合し、
run IDごとの実行を一度だけ受理する責任を持つ。** prepareだけのJSONは実行指示にしない。
受理・実行・修正確認・結果スキーマは [親レビュー入口](../review/README.md) を参照する。
同じprepareの再実行は同じrun ID、同じrequestの再送は同じ指示を使う。
既にレビュー対象となったPRの再レビューには `--reason-file` を付ける。別PRの初回レビューには不要。
PR open/push/CI完了ではこの入口もレビューも起動しない。

未指示 → `targets_fixed`（対象確定）→ `instructed`（指示済み）→ `result_received`（結果受領）→
`fixes_verified`（修正確認）→ `complete`（完了）の順。
後半は `review-state --run-id ID --state 状態 --evidence-file tmp/evidence.md` で証拠を残す。
受領と確認が同時なら `instructed` から `fixes_verified` へ直接進めてよい。指示前の確認や状態の後退は拒否する。
結果受領は成功判定ではなく、失敗・阻害指摘も受領して記録する。親が指摘と修正を照合し、
各PRのCIなどを確認した証拠が揃うまで完了へ進めない。CLIは#1219のmerge gateの代用ではない。

## 結果捕捉と親の判断

既存入口の結果・行動は非公開の判断journalへ自動捕捉する。
親の短い判断は `decide --spec`、状態と記録IDは `decision-log-status`、捕捉だけの再送は
`retry-decision-capture`、親ごとの明示的なGit公開は `publish-decision-log` を使う。
公開の単位・階層README・限定例外と復旧は[判断記録の公開仕様](../decision_logs/README.ja.md#git公開と閲覧)を参照する。
人間向けの入口は[親の結果・判断一覧](../../logs/parents/README.md)。親READMEの概要から活動の詳細へ辿る。
初回も `publish-decision-log --dry-run` で未準備記録を含む表示を確認する。プレビューの項目と制限は公開仕様を参照する。
取得元・未対応入口・判断spec・操作との結び付けは
[判断記録の仕様](../decision_logs/README.ja.md#捕捉元と未対応入口)を正本とする。
既存の要求・runner応答形式、成果受入や担当の権限は変更しない。

## 将来の割当側と実行条件

自動割当側は上記の候補検査と既存registryの排他を使い、対象親の実行根拠とruntime identityを結び付ける。
同じrepo・親番号/URL・環境由来ID・state・競合エラーを使い、子作成・実装・レビューの検査を飛ばさない。
判定agent・timerはこのCLIから起動しない。手動実装子の明示起動だけは `run-worker` が共通ランナーへ渡す。

CLIはJSONをstdout、エラーをstderrへ出し、成功0・設定/取得/整合エラー2で終了する。`run-worker` の実行後はランナーの終了値を返す。
ghは現在認証中のアカウント、GitHub書込みにはIssuesのwrite権限が必要。
親子ラベルの作成やsystemd操作は行わない。GitHubとローカルstateは単一トランザクションではないため、
非ゼロ終了後は同じ入力で照合・不足更新を行う。未完了を成功や新規起票に置換しない。
atomic write中にプロセスを強制終了して `.tmp` が残った場合は、稼働中プロセスが無いことと
registryの内容を確認して、その一時ファイルだけを人間の管理下で退避してから再試行する。

テスト: `bash .agents/ops/pytest_locked.sh tests/e2e/agents_ops/test_parent_session.py tests/e2e/agents_ops/test_parent_launch.py tests/e2e/agents_ops/test_manual_worker_runs.py tests/e2e/agents_ops/test_review_handoff.py tests/e2e/agents_ops/test_pytest_lock.py -q`。
fake gh・一時state・使い捨てgitを使い、本物のCodex/Claude/GitHub書込みは起動しない。
