# 終了時改善と全体受入

今回の成果の受入と、次回改善の採否・実行を分ける。候補を残しても元親のAcceptance・正式な子・依存を増やさない。
提案の公開・取り出しに失敗しても今回の受入を未達へ戻さず、`status.improvements` と `status.completion` で別々に確認する。
操作は既存の [parent CLI](README.md) と [issues API](../issues/README.md) を使う。

## 候補の正本

安定IDは `I-<元親番号>-<key>`。keyは親が選ぶ64文字以下の小文字英数字・ハイフンで、改名でも変えない。
元親の証拠用Issueの `parent-improvement:v1` 構造化コメント列が正本である。
[共通reader](README.md#機械証拠の保存共通reader) は旧親コメントも読み、revision、直前eventの参照、内容digestを検査する。
欠落・分岐・未知version・同じ配送keyの異なる内容を拒否する。子の「次の改善候補」欄は正本へのリンクだけを置く。
元親が終了しても候補の公開・判断・取り出しは元親の担当を再開しない。

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

今回の未達条件に関係する案は `unmet_acceptance` に `status.completion.requirements.criteria` のIDを記録する。
その候補の公開・承認と元親のcompletedは拒否する。自由文から達成の真偽を推定せず、親が証拠を確認する。
旧3フィールド（purpose/evidence/scope）の発見記録は読み続け、不足を同じkeyで補うまで公開しない。
候補の内容を更新するとrevisionが進み、以前の承認は失効する。2世代目も新しい候補・新しい人間承認が必要。

## 提案Issueへの列挙

親が終了時に候補を整理・記録した後、`improvement-propose` を明示実行する。
記録や `release --ended` だけでは自動起票しない。`release --ended` と完了した `workflow-report` の結果の `unpublished_improvements` で未掲載の候補を確認できる。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 improvement-propose
```

元親ごとに同時にopenな提案Issueは1件。ラベルは `issue:proposal`・`needs-human`・`enhancement`。
提案は作業対象ではなく、正式な親も子も持たない。役割の排他・各作業入口の除外は [issues規約](../issues/README.md#ラベル)を参照する。
出典は本文1行目の `<!-- parent-improvement-proposal:v1 parent=1216 generation=1 -->`。
初回作成と候補追加は正本の候補・判断・取り出しeventから本文を描画する。本文の手編集は判断に使わない。
各候補のID・revision・目的・観測と根拠・対象/対象外・受入条件・状態を列挙する。
期限はGitHubの作成日時から7日後（UTC）。候補追加や本文編集では延びない。
closeした提案は再openしない。後から記録した新しい候補は、次のgenerationの提案へ載せる。
終了した提案の候補の更新を、別generationへの再掲載として扱わない。

## 人間判断と実行

人間の会話での指示を、候補revisionと掲載先のopenな提案Issueへ結び付ける。
改善親の本文や補足確定は承認の入力にしない。候補の目的・範囲・対象外・受入条件・停止条件・根拠をsnapshotへ固定する。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 improvement-decide --candidate I-1216-result-access --action approve --spec tmp/human-decision.json
```

actionは `approve`（承認）/ `hold`（保留）/ `decline`（見送り）/ `revoke`（撤回）。単一候補のspec:

```json
{
  "schema_version": 1,
  "request_key": "human-20261113-1",
  "revision": 1,
  "previous_decision": null,
  "human_confirmation": {
    "quote": "この候補の記載した範囲を承認します",
    "source": "2026-11-13の人間との会話。候補 I-1216-result-access を確認"
  },
  "reason": "元発言を確認し、この候補の境界に対する承認として記録した"
}
```

revisionとprevious_decisionは正本コメントまたはstatusで確認する。2回目以降は直前のdecision_idを指定する。
同じrequest_key・内容の再送は不足した本文投影・停止処理を復旧し、後からの撤回を古い承認で上書きしない。
新しい承認は新しい元発言・request_keyが必要。候補の内容変更後は新しいrevisionを指定する。

1回の発言で複数候補を判断するときは `--candidate` を繰り返す。上記specのrevision/previous_decisionを次のmapへ置き換える。
共通の引用・出典・request_keyから候補ごとのeventを作り、全対象を検査してから追記する。途中失敗は同じspecで再送する。

```json
{
  "candidates": {
    "I-1216-result-access": {"revision": 1, "previous_decision": null},
    "I-1216-other": {"revision": 2, "previous_decision": null}
  }
}
```

人間とagentが同じGitHub loginを使うため、login・ラベル・チェックは人間の承認証明ではない。
保証範囲は既存agreeと同じで、親が人間の元発言を確認して引用・出典を記録する。AIの自己採用や元親の合意では代用しない。
判断後に提案本文を再描画する。提案がcloseした後の新しい判断・取り出しは拒否する。
ただし**取り出した候補への保留・見送り・撤回**は、既存parentを止めるため引き続き受け付ける。再承認はしない。

## 承認分の取り出しと復旧

人間が「承認した候補をparentにして」などと指示したときだけAIが実行する。自動取り出しのtimerはない。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1216 improvement-extract --candidates I-1216-result-access I-1216-other
# 取り出し結果のdecision_idsのいずれかを指定すると、記録された全候補の承認を取り込む
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1300 agree --improvement-approval DECISION_ID
```

指定集合は、その提案の現在承認済み・未取り出しの全候補と一致する必要がある。不足・余分を列挙して拒否する。
未判断・保留・見送り・撤回・期限切れや別の提案に属する候補は取り出さない。
新parentのラベルは `issue:parent`・`enhancement`。`needs-human` は付けない。
親テンプレートの節に全候補の承認snapshotと出典リンクをまとめ、出典目印に提案番号と全候補IDを保持する。
承認snapshotを起票後の自動背景追加で変えないよう、取り出したparentも本文補足の対象外とする。起動時の全承認照合は維持する。
取り出しeventを記録して提案本文を更新し、completedでcloseする。残った候補は「提案終了」として残す。

propose/extractは共有private stateへ固定draft・送信前receiptを永続化し、repo単位の改善操作ロックで直列化する。
既存の `create_issue_once` と全open/closedの出典索引を使い、応答喪失後も同じIssueで不足分だけを復旧する。
POST成功の可能性があり、索引で未観測なら `outcome_unknown` として停止し、再POSTしない。
同じ引数で再送する。receipt削除・key変更・タイトル検索では復旧しない。別ホストの独立state間の排他は提供しない。
取り出しが送信済みの提案は期限処理から外し、明示的なextract再送で起票結果・event・closeを収束させる。
送信後に承認が撤回された場合も、作成済みの同じparentとの照合とcloseだけを復旧する。元の承認参照を保存し、撤回後の実行は拒否する。

agree・create-child・delegate・run-worker（resume含む）・start/continue-worker・authorization・dispatcher・supervisorの起動直前で、全候補の現在の承認を照合する。
1候補でも撤回・保留・見送りになればparent全体の起動を拒否し、既存の停止要求経路を使う。他の候補だけを続けない。
停止未確認なら `stop.state=stopping` を返し、同じdecideを再送して回収を確認する。
実行合意は生成parent本文の保護対象要件から `issues.improvement.execution_terms` で変換する。
「目的・背景」→purpose、「全体のスコープ」→scope、「全体方針・合意事項」→behavior、「全体の完了条件」→acceptance。
「依存関係・進め方」と残りの保護対象節はbehaviorに加え、前文も保持する。必須5節の欠落・空欄は拒否する。
空白・チェック状態・進捗の体裁変更以外の要件変更は実行を拒否する。意味の同値性を推測しない。

## 期限切れcloseとunit導入

```bash
# 担当登録・session identityは不要。repo全体の提案を走査する
PYTHONPATH=.agents/ops .venv/bin/python .agents/ops/checkout_job.py --repo "$PWD" -- \
  .venv/bin/python -m parent --repo Motoki0705/tennis-lab improvement-expire --dry-run
# テスト・再現用の固定時刻
PYTHONPATH=.agents/ops .venv/bin/python .agents/ops/checkout_job.py --repo "$PWD" -- \
  .venv/bin/python -m parent --repo Motoki0705/tennis-lab improvement-expire --now 2026-11-21T00:00:00Z
```

作成から7日経過した、取り出されていないopenな提案をnot_plannedでcloseする。
期限判定にはGitHubの作成日時と正本に保存された出典を使い、本文の期限表示を信用しない。
失効した候補をcloseコメントに列挙し、承認済み・未取り出しを区別する。同じコメントを再投稿しない。
GitHub取得・更新に失敗すれば非0で終了し、stdoutのclosed/pending/errorsに部分結果を返す。
成功済みのcloseを戻さず、再実行で残りを処理する。

`parent/systemd/tennis-lab-improvement-expire.service` はoneshot、対応timerは日次・`Persistent=true`。
mainへmerge後に [unit installer](../install_units.sh) を使う（この操作だけでは有効化しない）。

```bash
spin ops install proposal-expiry
spin ops on proposal-expiry
```

unitと手動実行は [共通launcher](../lib/README.md#メインcheckoutと運用jobの排他) でcheckout lockを保持し、更新との競合を防ぐ。
unitはmain checkoutと既存unitと同じPATH、通常のgh認証設定を使う。`~/.bashrc` は読まない。
カスタムの認証環境が必要なら既存unitと同様にsystemdのuser環境へ明示設定する。
実ホストの導入・有効化・ラベル同期・代表フローの実GitHub受入は担当親が行う。

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
completedの前に[完了通知の契約](README.md#自由文の進捗連絡)で親へ固定completionを公開する。
finishは受入項目の機械記録として証拠用Issueへ保存し、通知本文を再掲しない。
自律は [workflow-report](development_workflow.md#待機完了担当終了) のcompleted入力version 2へfinishを含める。
実merge・レビュー・子の終了等の既存検査に加え、supervisorは当該job/担当generation/権限に結び付いたfinish receiptを照合する。
手書きの最終completed JSONだけでは完了しない。

取り出したparentの受入表示は、completionに保存した全decision_idの承認・要件と、現在の承認・本文を照合する。
同じ候補revisionでも新しい承認に過去の受入を引き継がず、元親statusのacceptedと取り出したparentの受入済み表示を外す。
過去のcompletionと公開済み終了コメントは保持する。文章の体裁だけの変更では受入を失効させない。

子なし `issue:parent` も [共通resolver](README.md#issue自身を実行する)で親自身の成果を照合する。
手動finishと自律completedは同じ実merge・受入・終了の証拠を要求し、空の子一覧だけで完了にしない。
候補の発見元に架空の子は作らず、独自のresolverや完了fallbackは置かない。

## 判断ログと互換

候補観測・切り出し理由・公開/取り出し要求と結果・会話判断は [既存判断journal](../decision_logs/README.ja.md) へ接続する。
`improvement` が返す公開decision_idは `improvement-propose --decision-id` へ渡せる。
同じ判断参照の再送はretry_publicationとして前回結果へ接続する。正本の理由を別ログへ再入力しない。
終了した元親のjournalは別sessionから更新せず、取り出したparentのagreeで全承認を観測した結果と元コメントへの参照を記録する。

既存の候補記録・旧published/decision eventは共通readerで読み続ける。旧方式の改善親を見つけた場合は自動移行せず担当親へ返す。
運用開始前にmainの新しいCLI・dispatcherへ切り替え、旧runを既存の停止・回収手順で確認する。
helpとfake GitHubの起動拒否・復旧テストを確認し、実機受入は人間承認の範囲で1つの代表フローを行う。
