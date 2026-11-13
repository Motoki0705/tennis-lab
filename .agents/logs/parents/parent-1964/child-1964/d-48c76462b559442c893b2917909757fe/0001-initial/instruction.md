# #1964 issue:proposal 改善提案Issueと承認済み候補の取り出しの実装

担当Issue: https://github.com/Motoki0705/tennis-lab/issues/1964 （親自身を作業対象とする未分割Issue。子Issueは作らない）
worktree: `.`（branch `codex/issue-proposal-label`、origin/main `b734cb96a` 起点）

要件の正本は #1964 本文の「全体のスコープ」「全体方針・合意事項」「全体の完了条件」。最初に本文を全部読むこと。
現行方式の正本は `.agents/ops/parent/improvements.md`・`improvements.py`・`parent/cli.py`・`issues/improvement*`（あれば）・関連テスト
（`tests/e2e/agents_ops/test_parent_improvements.py`・`test_improvement_recovery.py`・`test_parent_completion_records.py`）。
直前の #1878（PR #1942）で `issue:standalone` を役割ラベルとして追加した。役割ラベルの定数・排他検査・`issues check` の問題コード・検索例は、その仕組みに揃える。

今回のworkerの担当は**コード・規約・skill・テンプレート・unit定義・テストの変更とPR作成まで**。
次は親が行うので、あなたは行わない:
- GitHub上のラベル作成（`sync_github_labels.py --apply`）
- 実GitHubでの提案Issueの作成・承認・取り出し
- systemd unitの実機導入
- Issueのclose・コメント投稿、merge

## 親が決めた設計（この方針で実装する。不都合があれば理由と代案を最終報告で返す）

1. **ラベル**
   - `.github/labels.yml` の `hierarchy` に `issue:proposal` を追加する。色は親子・standaloneと区別できる色にする。説明は「未判断を含む改善候補を元親ごとに列挙する改善提案Issue。作業対象ではない（.agents/ops/parent/improvements.md）」程度にする。
   - 定数 `PROPOSAL_LABEL` は `issues/format.py` に置いて公開する。文字列の直書きを増やさない。
2. **役割の排他・除外**
   - `issues check` は、proposal と parent/child/standalone の混在、および正式なparent/childrenとの矛盾を新しい問題コードで検出する。proposalの本文に親子の必須節は要求しない。
   - `render_spec` は任意labelsへのproposal混入を拒否する。
   - proposalのIssueは次の入口で作業対象にしない（拒否する、または列挙から除外する）:
     - 親CLIの register・agree・create-child・delegate・run-worker・start-worker
     - dispatcherの候補判定
     - supervisor
     - issue_supplement（本文補足workflow）
     - project_progress（Project表示）
     - ops-reportの列挙・triage
   - 各入口で既存の parent/child/standalone の判定が変わらないことをテストで固定する。
3. **提案Issue（`improvement-propose`）**
   - `improvement-publish` を置き換える新コマンド `improvement-propose`（元親 `--parent N`）を作る。元親に記録済みの候補（取り出し済み・closeした提案に載った候補を除く）をopenな提案Issueへ載せる。
   - openな提案がなければ作成し、あれば本文を正本から再描画する。元親ごとに同時にopenな提案は1件だけ。
   - ラベルは `issue:proposal`・`needs-human`・作業種別（現行の `enhancement` 相当）。parent・childラベルや正式な親子関係は付けない。
   - 出典目印は本文1行目に置く。例: `<!-- parent-improvement-proposal:v1 parent=N generation=K -->`。closeした提案のあとに作る提案は generation を進める。
   - 作成は既存の `create_issue_once`（固定draft・送信前receipt・出典索引・`outcome_unknown`）を再利用し、重複Issueを作らない。
   - 本文は正本（候補記録＋判断記録＋取り出し記録）からの決定的な投影にする。描画する項目:
     - 各候補: 候補ID、revision、目的、観測・根拠、範囲・対象外、受入条件、状態（未判断／承認／保留／見送り／撤回／取り出し済み → #N／提案終了／期限切れ）
     - 提案の期限日時（作成時刻＋7日、UTCとJSTの表記はREADMEの既存慣行に合わせる）
     - 判断方法の案内（会話で親AIに伝える。本文の編集やラベル操作は判断にならない）
   - 本文の手編集は判断に使わない。
   - いつ propose を実行するかは parent-session skill と improvements.md の手順に書く。目安は、親が終了時に候補を記録した後。`release --ended` などの既存の終了経路で、提案Issueに未掲載の候補があれば、それを結果に示す（自動起票はしない）。
4. **会話判断（`improvement-decide`）**
   - 判断の対象を「候補revision＋openな提案Issue番号」に変える。改善親番号と改善親本文の固定、および「補足確定までapproveできない」制約は外す。
   - 承認は、承認時の候補snapshot（目的・範囲・受入条件などの requirements）を固定する。
   - 1回の発言で複数候補を承認できるようにする。`--candidate` を複数指定するか、specに候補一覧を持たせ、同じ引用・出典・request_keyから候補ごとの判断eventを作る。再送は冪等にする。後から行われた撤回を古い承認で上書きしない、という既存の規則は維持する。
   - closeした提案の候補への判断は拒否する。
   - 判断後は提案Issueの本文を再描画する（失敗時は再送で収束する）。
5. **取り出し（`improvement-extract`）**
   - 引数は `--parent <元親> --candidates I-a I-b ...`。
   - 指定候補がすべて現在有効な承認を持つときだけ実行する。さらに、指定集合はその提案の承認済み・未取り出しの候補全体と一致しなければならない。一致しなければ不足・余分を示して拒否する（承認分をまとめて1件、というユーザー決定）。
   - 未判断・保留・見送り・撤回・取り出し済み・closeした提案の候補を含めたら拒否する。
   - 新しいparentの起票:
     - ラベル: `issue:parent` と作業種別。`needs-human` は付けない。
     - 本文: 親テンプレートの節構成で、承認snapshotから描画する。全候補の目的・範囲・受入条件・出典リンクと、提案Issueへのリンクを含める。
     - 出典目印: 全候補IDと提案Issue番号を持つ。
     - 起票は `create_issue_once` を使う。
   - 起票後、提案Issueに「取り出し済み → #N」を描画してcloseする（state_reason completed）。取り出されなかった候補は「提案終了」になる。
   - 起票とcloseの途中で失敗しても、再実行で同じparentを照合し、closeまで進める。
   - 実行承認: 取り出したparentを `agree --improvement-approval` で承認群から実行できるようにする。取り出し記録から承認decision_id群を引けるなら、それを使ってよい。
     - 現行の起動直前の承認照合（agree・create-child・delegate・run-worker・start/continue-worker・authorization・dispatcher・supervisor）を、複数承認に拡張する。
     - 取り出し後に1候補でも承認が revoke/hold/decline されたら、そのparentの起動を拒否し、既存の停止要求経路で停止する。
   - 実行合意（purpose/scope/behavior/acceptance）は、生成したparent本文から既存の `execution_terms` と同じ規則で変換する。
6. **期限切れclose（`improvement-expire`）**
   - repo全体の提案Issueを走査する。作成時刻＋7日を過ぎ、取り出し記録がないopenな提案を not planned でcloseする。
   - closeコメントには、失効した候補（承認済みで未取り出しのものを区別する）を列挙する。
   - 期限の判定は、GitHubのIssue作成時刻と出典目印から行う。本文の期限表記は判定に使わない。
   - 担当登録を必要としない入口にする（例: `python -m parent improvement-expire`）。`--now`（テスト用の固定時刻）と `--dry-run` を持たせる。
   - GitHubの取得・更新に失敗したら非0で終了し、部分的に成功した結果を出力する。再実行で同じ結果に収束させる（冪等）。
   - 日次timer・serviceを追加する（例: `.agents/ops/parent/systemd/tennis-lab-improvement-expire.{service,timer}`）。`OnCalendar` は日次、`Persistent=true`、oneshot。
     - `install_units.sh` で導入されるようにし、該当READMEに書く。
     - systemd環境は `[local path]` を読まないため、gh認証などで必要な環境を既存unitと同じ方法で与える。
7. **移行・互換**
   - `improvement-publish` を削除し、新方式へ置き換える。
   - 既存の候補記録・判断記録（改善親を対象にした旧event）は読み続ける。旧方式で作られた改善親は0件の見込み。全open/closedを出典目印 `parent-improvement-source` で確認し、存在すれば最終報告で知らせる（自動移行しない）。
   - decision_logs（`parent_commands.py`・READMEの日英中）の対応表も新コマンドへ更新する。
8. **文書**
   - 次を新方式に揃える:
     - `improvements.md`
     - `parent/README.md`
     - `.agents/skills/parent-session/SKILL.md`（「独立した新親の起票は `improvement-publish`」の行など）
     - `issues/README.md`（役割表・ラベル軸・検索例に proposal を追加。提案は作業対象ではないこと）
     - decision_logs README
   - 同じ事柄を2か所に書かない。

## テスト

[tests README](tests/README.md#テスト作成方針) に従って設計し、既存のfake GitHubとテストを再利用する。少なくとも次を検証する:
- ラベルmanifestと定数、render_specの拒否、`issues check` の proposal 混在・正式関係との矛盾、既存役割の非退行
- 各作業入口がproposalを対象にしない
- propose: 初回作成・候補追加で同じIssueを再描画・応答喪失から重複なしで復旧・本文手編集が判断にならない・close後は新generation
- decide: 候補revision固定、内容変更後の承認拒否、1発言の複数承認、再送の冪等性、撤回を古い承認で上書きしない、close後の判断拒否
- extract: 指定集合と承認集合の不一致・未承認・取り出し済みの拒否、POSTのラベル（parentあり・needs-humanなし）・本文・出典目印、応答喪失・再送・途中失敗からのclose完了、重複parent 0件
- 取り出したparentの agree→delegate の代表フロー、取り出し後に1候補を撤回すると起動拒否・停止要求になる
- expire: 固定時刻で期限前は何もしない、期限後はclose、再実行で変化なし、GitHub失敗で非0、取り出し済みの提案は対象外
- 旧形式の判断記録を読み続けられる

関連テストを実行する（例: `.venv/bin/python -m pytest tests/e2e/agents_ops -k "improvement or issue_format or parent_session or dispatcher or supplement or progress or triage" -n 4 -q`）。広い範囲は `.agents/ops/pytest_locked.sh` の慣行に従う。失敗が変更と無関係なら、origin/mainで同じテストを比べて示す。

## 成果

- コミットは pre-commit を通す。
- PRを作る。PR本文は日本語で書き、`Closes #1964` は付けずに `Refs #1964` とする（ラベル反映と実機受入が残るため）。
- pushごとに、対象HEADのCI `Python` の結果を確認して最終報告へ含める。
- 作業時間の目安は最大4時間。上記の設計とテストが揃うまで止めない。止めてよいのは、要件どうしの矛盾や、既存契約と両立しない設計点が見つかった場合だけ。そのときは根拠と選択肢を最終報告で返す。

最終報告には次を含める:
- PR番号・HEAD
- 変更ファイルの要約
- #1964 の各完了条件に対する、コードで担保した部分の証拠
- 実行したテストと結果
- CI結果
- 旧改善親の有無の確認結果
- 未確認事項
