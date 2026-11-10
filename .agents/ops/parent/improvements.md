# 終了時改善と全体受入

今回の成果の受入と、次回改善の採否・実行を分ける。候補を残しても元親のAcceptance・正式な子・依存を増やさない。
改善の起票や公開が失敗しても今回の受入を未達へ戻さず、`status` でそれぞれの状態を確認する。
操作は既存の [parent CLI](README.md)、共通の [issues API](../issues/README.md) を使う。

## 候補の正本

安定IDは `I-<元親番号>-<key>`。keyは親が選ぶ64文字以下の小文字英数字・ハイフンで、改名・再提案でも変えない。
元親の `parent-improvement:v1` 構造化コメント列が唯一の正本である。候補revision、直前eventの参照、内容digestを検査し、
欠落・分岐・未知version・同じ配送keyの異なる内容を拒否する。子の「次の改善候補」欄は正本へのリンクだけを置く。
元親がcloseしてもコメントを参照でき、同じ候補の補足・起票・人間判断の再送は元親の担当・実装を再開しない。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 improvement --key result-access --spec tmp/improvement.json
# 子で発見した場合だけ --child 1217 を付ける
```

`improvement.json` は次の全フィールドを持つ。文章は非空、acceptanceは検証可能な非空配列。

```json
{
  "schema_version": 1,
  "purpose": "保存済み結果を取得する操作を減らす",
  "observation": "元の受入は完了したが、取得操作が重複していた",
  "evidence": "PR #1250 の検証記録で同じ取得操作を2回確認した",
  "judgment": "今回必須の不具合ではなく独立した使い勝手の改善として切り出す",
  "scope": "保存済み結果の取得入口と失敗表示",
  "out_of_scope": "元の実装・モデル・レビュー方針・GPU実行",
  "behavior": "1回の明示操作で保存済みの結果を取得できる",
  "acceptance": ["再取得で実装workerを起動しない", "取得失敗を表示して保存済み結果を保持する"],
  "stop_conditions": "記載した受入条件を満たす。追加要件は別途判断する",
  "unknowns": "運用上の削減時間は未測定",
  "expected_effect": "重複した管理操作を減らす",
  "unmet_acceptance": []
}
```

今回の未達条件に関係する案なら `unmet_acceptance` に `status.completion.requirements.criteria` のIDを記録する。
その候補の起票・承認と元親のcompletedは拒否する。自由文から達成の真偽を推定せず、親が証拠を確認する責任を持つ。
旧3フィールド（purpose/evidence/scope）のspecは発見記録として保持し、返されたkeyで不足を補うまで起票できない。
記録・整理だけで別issueを作ることはない。候補の更新は承認を引き継がない。

## 起票と復旧

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 improvement-publish --candidate I-1216-result-access
# 同目的の既存issueへ集約する場合
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 improvement-publish --candidate I-1216-result-access --existing 1300
```

新規作成は親テンプレートからrenderし、**POSTのlabelsに `issue:parent` と `needs-human` を含める**。
出典は本文1行目の `<!-- parent-improvement-source:v1 parent=1216 key=result-access -->`。
`ai-proposed`・`ops-report-proposal`・`parent-task:v1` は流用しない。正式親・依存は追加しない。
タイトルでは同一性を判断しない。人間/親が同目的を照合して `--existing` へ渡す。既存open・closed/completed・
closed/not_plannedの本文・ラベル・親子関係・状態は維持する。参照の追加は既存issueの実行権限を拡張せず、既存の合意経路を使う。
すでにこの候補から作成したissueも、改名・close・見送り後に別issueへ複製しない。

共有private stateへ固定draft・送信前receiptを永続化し、repo単位の操作ロックで同時再送を直列化する。
全open/closedの出典索引から作成済みissueを照合し、リンク記録だけが欠けた場合はその部分を補う。
POSTが成功した可能性があるのに応答が失われ、索引にもまだ見えない場合は `outcome_unknown` で停止する。
同じ候補IDで再送して照合する。receipt削除・key変更・タイトル検索による再POSTで復旧しない。
同じrepoの操作は同じstate/lockを共有する。別ホストの独立stateからの同時書込みの排他は提供しない。

再提案は同じ候補key・同じissueで境界と根拠を更新し、新しい人間判断を記録する。本文の新しい要件も人間に確認してもらう。
issueのreopenや `needs-human` の解除は、この操作の副作用として行わない。

## 人間判断と実行

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 improvement-decide --candidate I-1216-result-access --action approve --spec tmp/human-decision.json
```

actionは `approve`（実行承認）/ `hold`（保留）/ `decline`（見送り）/ `revoke`（承認撤回）。specは次の形式。

```json
{
  "schema_version": 1,
  "request_key": "human-20261111-1",
  "revision": 1,
  "previous_decision": null,
  "human_confirmation": {
    "quote": "この候補の記載した範囲で実装を進めてください",
    "source": "2026-11-11の人間との会話。候補 I-1216-result-access と改善親 #1300 を確認"
  },
  "reason": "元発言を確認し、この候補の境界に対する実行承認として記録した"
}
```

revisionとprevious_decisionは最新status/正本コメントで確認する。2回目以降は直前のdecision_idを指定する。
同じrequest_key・内容の再送は不足した停止処理だけを復旧し、後から行われた撤回を古い承認で上書きしない。
新しい承認は新しい元発言・request_keyが必要。対象issueの補足追加が確定するまでapproveできない。

人間とagentが同じGitHub loginを使うため、loginやラベルを人間の承認証明にはしない。
**保証範囲は既存agreeと同じ**で、親が人間の元発言を確認して引用・出典を記録する。別credential・署名・人間性の認証は提供しない。
AIの自己採用、元親の合意、自律実行根拠、`needs-human` の削除は専用判断eventの代わりにならない。

未承認でも改善親の `register` / `handoff --accept` は可能で、議論・判断の記録を進められる。
実装は `agree --improvement-approval <decision_id>` で承認を取り込み、既存のcreate-child/delegate/run-worker経路へ渡す。
agree・create-child・delegate・run-worker（resumeを含む）・start-worker/continue-worker・実行authorization・
dispatcher候補判定・supervisorの起動直前で、正本の現在の承認を照合する。
dispatcherに新たな自動割当経路は作らず、既存の自律着手条件に従う。承認記録で `needs-human` は外さない。

承認は候補revision・改善親番号・本文の要件を固定する。空白・チェック状態・進捗節の変化は許容し、
目的・範囲・受入条件・未知の追加節・コードなどの変化は新しい判断まで拒否する。意味の同値性をLLMで推定しない。
保留・見送り・撤回は起動直前の検査と同じロックで停止要求を保存し、既存runner/supervisorで対象の停止・回収を照合する。
未確認なら `stop.state=stopping` を返し、同じdecideを再送して復旧する。停止確認前に承認し直しても起動できない。
2世代目も新しい候補・新しい承認が必要で、前世代の権限は継承しない。

## 今回の受入と候補整理

合意コメントにあるagreementと本文snapshotの受入チェック項目を使う。別の受入台帳へ移さない。
`status.completion.requirements` の `basis_ref` と全criteriaのIDを読み、次のfinishを作る。
例の項目数は固定ではない。すべてのIDを一度ずつ、根拠とともに記録する。

```json
{
  "schema_version": 1,
  "state": "completed",
  "basis_ref": "statusで取得した値",
  "acceptance": [
    {"id": "agreement", "state": "met", "evidence": "合意の受入をPR #1250 と検証記録で確認"},
    {"id": "ac-1", "state": "met", "evidence": "当該チェック項目の実測と成果物への参照"}
  ],
  "remaining_required": [],
  "improvements": {"reviewed": true, "candidates": ["I-1216-result-access"]},
  "summary": "今回の全条件を受入済み。取得入口の改善は独立した人間判断へ渡す"
}
```

候補なしは `reviewed: true, candidates: []`。登録済みの全候補を一度ずつ参照し、候補を黙って落とさない。
`state: incomplete` では項目をunmet/unverifiedにできるが、completedは全項目met＋非空根拠とremaining_required空が必要。
本文の受入項目が合意snapshotから変わった場合は合意を確認し直す。機械的に基準を特定できない既存親は
requirements.available=falseとなり、finishを拒否する。一括移行や推測で埋めない。

手動は `release --ended --spec tmp/finish.json`。従来の `--note-file` は受入未確認の担当終了として維持する。
自律は [workflow-report](development_workflow.md#待機完了担当終了) のcompleted入力version 2へfinishを含める。
実merge・レビュー・子の終了等の既存検査に加え、supervisorは当該job/担当generation/権限に結び付いたfinish receiptを照合する。
手書きの最終completed JSONだけでは完了しない。

子なし `issue:parent` でも手動finish・候補記録・起票・判断はできる。自律×子なし親の実行対象resolverと
completedまでのE2E受入は **#1804待ち**。この機能は架空の子やfallback resolverで補わない。

## 判断ログと導入順

元親担当中の候補観測・切り出し理由・起票要求/結果は [既存判断journal](../decision_logs/README.ja.md)へ捕捉する。
`improvement` が返す公開decision_idを `improvement-publish --decision-id` へ渡すと、その判断と起票を関連付けられる。
同じ候補・宛先・判断参照の再送は前回結果につながるretry_publicationとして捕捉し、同じ計画を二重に要求しない。
理由は候補の正本を参照し、`decide` 等へ同じ文章を再入力しない。人間判断の正本も候補コメントである。
終了済み元親のjournalを別sessionから公開・更新する経路は追加しない。改善親の担当がagreeへ承認を取り込んだ時点で、
改善親graphに「承認を観測した結果」と元コメント参照を記録する。未判断・見送り・承認済み未起動・実行失敗は成功と区別する。
#1865/#1871の捕捉・公開用変換を利用し、**#1876の表示連携の受入はmerge後**に行う。

最初の新出典publishの前に、この変更を含むmainのopsへdispatcherサービスと手動CLIの両方を切り替える。
古いopsを使った親/workerの委任を止め、稼働中のrunは既存の停止・回収手順で確認してから新しいopsで再開する。
dispatcherのinstall/運用とprivate stateの保持は [dispatcher README](../dispatcher/README.md)に従う。
新しい入口のhelpと、fake GitHubでの起動拒否テストを確認してからpublishを使う。実GitHubへの試験大量起票や過去issueの一括移行はしない。
