# 親子issueの規約と共通ツール

タイトル・ラベル・本文・親子関係・進捗記録の正本。手動起票、親セッションによる分割（#1217）、
レポート提案からの起票（#1223）はこの規約と共通APIを使う。親セッションの割当・要件合意・
レビュー・マージの実装はここでは扱わない。

## 役割とテンプレートの選択

| ラベル | 意味・付与条件 |
|---|---|
| `issue:parent` | 全体の目的・合意・進捗を管理する親。分割前にも付ける |
| `issue:child` | GitHubの正式なsub-issueとして直属の親を持ち、具体的な1タスクを担当する子 |
| `issue:standalone` | 親にも子にも属さず、単独の作業単位として扱うIssue。正式な親も子も持たず、管理上も親として扱わない作業Issueに付ける |

親と子を兼ねるissueには `issue:parent`・`issue:child` の両ラベルを付ける。
`issue:standalone` は両ラベルと排他。未分割の親や全子完了後の親は、子が0件でも親のまま扱う。
他issueへの参照・依存関係・発見元があることだけで子とは判定しない。
`ops-report` のレポート容器とPRにはstandaloneを付けない。
別軸のラベルは役割区分の代わりにならず、ラベルから親番号を推測しない。
役割ラベルのない既存issueをラベル欠如だけで単独と断定せず、作業対象になった時に段階適用する。

この分類ラベルの付与から実行承認・担当予約は生成しない。既存の実行根拠・停止・レビュー・merge条件を維持し、
`needs-human` は引き続き人間の承認待ちを意味する。

| 利用場面 | テンプレート |
|---|---|
| 全体の構想・複数タスクの統括 | [親issue](../../../.github/ISSUE_TEMPLATE/06_parent.md) |
| 親の合意範囲から分割した1タスク | [子issue](../../../.github/ISSUE_TEMPLATE/07_child.md) |
| 親も子も持たない単独タスク | [バグ](../../../.github/ISSUE_TEMPLATE/01_bug.md)・[機能](../../../.github/ISSUE_TEMPLATE/02_feature.md)・[調査](../../../.github/ISSUE_TEMPLATE/03_research.md)。作業種別と `issue:standalone` を付ける |
| 研究ループ | [研究テーマ](../../../.github/ISSUE_TEMPLATE/04_research_theme.md)は親、[研究サイクル](../../../.github/ISSUE_TEMPLATE/05_research_cycle.md)は子。共通項目に研究固有の項目を加える |

単独タスクを分割するときは親へ、既存の親へ編入するときは子へ[明示的に移行](#単独から親子への明示移行)する。
研究の指標・評価条件・合意・知見の置き場やclose判断は[research-loop](../../skills/research-loop/SKILL.md)に従う。

## ラベル

名前・色・説明と軸ごとの一覧の正本は [.github/labels.yml](../../../.github/labels.yml)。
意味と付与主体はここで定義する。ラベルの作成・更新・削除は管理者が下記の同期CLIで行う。

| 軸（マニフェストのキー） | 意味 | 付与主体 |
|---|---|---|
| 階層（`hierarchy`） | 親・子・単独の役割（上表） | 起票者・担当親・テンプレート・親CLI・triage。`issues check` で正式な関係と照合する |
| 作業種別（`work-type`） | バグ修正・機能改善・調査研究・文書・追補 | テンプレート、起票者・親CLIの明示指定、issue-labelerのタイトル先頭prefix |
| 起票元（`origin`） | `ai-proposed` は運用レポートの採用提案からの起票、`agent:*` は起票agent | 前者はレポートtriage、後者は起票agent。通常のAI作業・親CLIの子起票には `ai-proposed` を付けない |
| 対象領域（`area`） | 対象モジュール・CI・テスト・依存・外部コード・Python・知見 | issueは起票者・親CLIの明示指定だけ。PRは `.github/labeler.yml` が変更ファイルから付け、Dependabotは設定で指定する |
| 状態（`status`） | `needs-human` は人間の承認待ち、`ops-report` と `parent-evidence` は記録容器、`supplement:*` は[本文補足追加の確定状態](#親issueの補足追加状態) | 承認を必要とする仕組み・担当者、各記録CLI、補足追加workflow（skipは人間） |

issue-labelerは [.github/issue-labeler.yml](../../../.github/issue-labeler.yml) のタイトル先頭prefixから
作業種別だけを付け、本文を読まない。対象領域をタイトルからも推測しない。
親子の識別名prefixは作業種別を意味しないため、テンプレートや明示指定で補う。

同期と既存open issueの誤ラベル解除は次で計画を確認する。`--repo` 省略時は `gh repo view` を使う。

```bash
.venv/bin/python scripts/sync_github_labels.py sync --repo Motoki0705/tennis-lab --prune
.venv/bin/python scripts/sync_github_labels.py strip-auto-labels --repo Motoki0705/tennis-lab
```

既定はdry-run。内容を確認後、各コマンドに `--apply` を加えて実行する。
`sync` は不足ラベルの作成・色と説明の更新を行い、`--prune` 指定時だけ正本にないラベルを削除する。
色の英字の大小は区別しない。名前の変更・別名の移行は行わない。
`strip-auto-labels` はPRとclosed issueを除外し、`area` 軸の現在のラベルについて全ページのissue eventsを取得する。
最後の `labeled` イベントのactorが `github-actions[bot]` のものだけを解除する。
人間・agentによる付与は残し、イベント・actorを特定できないものも理由を表示して残す。
全対象の取得に成功してから書き込みを始め、解除直前にもissueの状態と全イベントを再確認する。
取得失敗を空結果へ置き換えず、非ゼロで終了する。

### 人間向けの検索

単独タスク一覧:

```text
repo:Motoki0705/tennis-lab is:issue is:open label:"issue:standalone"
```

単独の調査タスク:

```text
repo:Motoki0705/tennis-lab is:issue is:open label:"issue:standalone" label:research
```

親・子・単独のいずれにも未分類の確認候補:

```text
repo:Motoki0705/tennis-lab is:issue is:open -label:"issue:parent" -label:"issue:child" -label:"issue:standalone" -label:ops-report
```

最後の検索は確認候補の抽出であり、全件にstandaloneを自動付与する根拠にはしない。

## 証拠用Issueの除外

`issues.evidence.is_evidence_issue` は `parent-evidence` ラベルまたは本文先頭の
`parent-evidence` マーカーで保存容器を識別する。open/closed、親子ラベルの誤付与、
未知versionでも作業候補にしない。`require_work_issue` は明示対象や正式関係に混入した容器を拒否する。
結び付きの認証・作成・履歴の契約は [parent reader](../parent/README.md#機械証拠の保存共通reader) が正本。

| 経路 | 除外境界 |
|---|---|
| issues API・親CLI | `family` のroot/子、`GitHub.issue` の明示対象を拒否。`issue_rows` は列挙から除外。作業draftのラベルとしても指定不可 |
| dispatcher・提案整理 | 全open/all列挙から除外。`is_container`/guardも拒否し、正式関係・依存への混入はエラー |
| 補足追加・issue-labeler workflow | 自動イベントのjob条件で除外。手動入口は最新Issueを再取得し、書込/agent起動より前に除外 |
| ops-reportの履歴・triage・採否収集 | `shell.gh_list` がラベルと本文を取得して除外。organizationは上の共通列挙を使用 |
| project_progress | 正式treeの読取・Projectへの登録/同期対象で拒否。別Project itemは成果の分母に入れない |
| worker_status・board | GitHub snapshot/担当同期は共通 `GitHub.issue/family` から読む。ローカル表示はその入口で登録された親registryだけを使用 |
| herdr_labels | GitHub Issue列挙を行わず、共通の親CLIが検証して担当登録したregistryを使用 |
| ラベル整理CLI | open Issueの列挙から除外 |

旧 `weekly-report` のunit/コードは撤去済み。現行はops-reportであり、再導入を防ぐ検査と移行手順も
[ops-report](../ops-report/README.md) にある。
証拠コメントや既存permalinkを明示的に読む汎用の資料取得APIは、作業候補化せず閲覧できる。

## 親issueの補足追加状態

判定の正本は `issues.supplement_state`。ローカルregistryやworkflowの終了状態から本文の確定を推測しない。
有効化の正本はrepository variable `ISSUE_SUPPLEMENT_ENABLED_SINCE`（UTCの `YYYY-MM-DDTHH:MM:SSZ`）。
未設定のHTTP 404だけを無効とし、403・5xx・通信失敗・不正な応答や日時は例外にする。
設定とGitHub上のラベル作成は管理者が別途行う。

有効かつ現在 `issue:parent` を持ち、GitHubの `created_at` が有効化時刻**以降**のissueだけが対象。
それ以外は `not_target`。対象で確定ラベルがなければ `pending`、あれば以下の状態になる。
確定ラベルが2種類以上付いていたら、非対象のissueでも例外にする。

| 公開定数 | ラベル | 状態・意味 |
|---|---|---|
| `SUPPLEMENT_DONE_LABEL` | `supplement:done` | `done`: 補足追加完了 |
| `SUPPLEMENT_FAILED_LABEL` | `supplement:failed` | `failed`: 補足追加に失敗し、原文で確定 |
| `SUPPLEMENT_SKIPPED_LABEL` | `supplement:skipped` | `skipped`: 人間が補足追加を省略 |

### Python API（workflowとの共有契約）

import rootは `.agents/ops`。workflowも次の公開名をそのまま使う。

```python
from issues.cli import GitHub
from issues.supplement_state import (
    SUPPLEMENT_DONE_LABEL,
    SUPPLEMENT_FAILED_LABEL,
    SUPPLEMENT_SKIPPED_LABEL,
    SUPPLEMENT_FINAL_LABELS,
    SUPPLEMENT_ENABLED_SINCE_VARIABLE,
    read_enabled_since,
    supplement_state,
)

gh = GitHub("owner/repo")
enabled_since = read_enabled_since(gh)
issue = gh.get("repos/owner/repo/issues/123")
state = supplement_state(
    [label["name"] for label in issue["labels"]],
    issue["created_at"],
    enabled_since,
)
```

- `read_enabled_since(gh) -> str | None`: `repo` と `get(endpoint) -> object` を持つ読み取りclientを受け取る。
  共通 `issues.cli.GitHub`（`parent.github.GitHub` も可）が `gh api --include --method GET` で
  `repos/{repo}/actions/variables/ISSUE_SUPPLEMENT_ENABLED_SINCE` を取得する。
  `GitHubHTTPError.status == 404` のときだけ `None`、設定済みなら検証したUTC文字列を返す。
- `supplement_state(labels, created_at, enabled_since) -> SupplementState`: `labels` はラベル**名の文字列**のiterable。
  RESTのラベルobjectそのものは渡さない。日時はUTC秒形式の文字列、無効の場合だけ `enabled_since=None`。
  不正な日時・確定ラベルの競合は `ValueError`。戻り値の型 `SupplementState` は上記5状態の `Literal`。
- `SUPPLEMENT_FINAL_LABELS` は3つの確定ラベルの `frozenset[str]`、
  `SUPPLEMENT_ENABLED_SINCE_VARIABLE` はrepository variableの名前。
  定数・型・`read_enabled_since` は `issues` packageからも公開する。
  `from issues import supplement_state` は関数ではなく同名の**module**を返す。

親CLIのgateと待機は [parent README](../parent/README.md#補足追加の確定待ち)、
自律着手の保留と再照合は [dispatcher README](../dispatcher/README.md#判断結果と保留コメント)を参照する。

## タイトル

- 親: `[開発ループ] 実装と全体改善を循環させる`
- 子: `[開発ループ #1216] 親子issueの書式を統一する`
- 子がさらに親を兼ねる場合: `[開発ループ #1216] [書式] 親子issueの書式を統一する`
  その直属の子: `[書式 #1225] 本文の検査を実装する`

親の識別名は目的を表す短い語句を担当親が決める。同じ親では固定し、番号だけ、改行、`[` / `]` / `#`
を含む名前は使わない。同名の親があっても番号で区別する。角括弧の後には何を達成するかを具体的に書く。
子の表示タイトル先頭に提案IDや作業種別のprefixを重ねない。

親の要約だけを改名した場合、直属の親のチェックリストにあるそのタスク名を更新する。
親の**識別名**を変更した場合は直属の全子のprefixも生成し直す。さらに子を持つ場合も、その子自身の
識別名は変えない。子の要約の改名は親のチェックリストにも反映する。
親の付替えでは旧親・新親の担当が調整し、正式な関係、タイトル、本文参照、両親の一覧を揃える。
識別名やタイトルの変更で提案IDを変更しない。

## 本文

見出しは以下の表記の `##` を使う。必須節は1回だけ置き、空欄やコメントだけで済ませない。
追加の研究固有項目などは別の節として残せる。子が親を兼ねる場合は両方の必須節を持つ。

| 種別 | 必須の節・行 | 任意の節 |
|---|---|---|
| 親 | 目的・背景 / 全体のスコープ / 全体方針・合意事項 / 全体の完了条件 / 子タスクのチェックリスト / 進捗概要 / 依存関係・進め方 | 未決事項・人間の判断が必要な点（存在すれば必ず書く） |
| 子 | 冒頭の `親issue: #番号` / このタスクの目的 / 具体的な変更内容・対象範囲 / 対象外 / 依存関係 / 作業状況・停止事項 / Acceptance checklist | 成果物・関連PR / 次の改善候補（該当すれば必ず書く） |

親は「なぜ・全体で何を達成するか」、子は「このタスクでどこをどう変え、どう確認するか」を書く。
親の合意内容を子に複製せず親を参照し、担当agentが開始できる具体性は子に持たせる。
スコープには対象と対象外を書く。依存なし・成果物未作成・未合意・未決も明示し、確定や完了と混同しない。
今回必須ではない改善候補は目的・根拠・想定範囲を子に記録し、受入条件に自動追加しない。

子のAcceptance Criterionはチェック行に1つの条件を書き、直後に空行を挟まず、半角2スペースで
次の3行を添える（各行は1行の文章）。検証コマンドや証拠への参照を具体的に書く。

```markdown
- [ ] 親子情報の矛盾を検出できる
  文脈: 一覧と本文で同じ親を指す必要がある。
  下流への影響: 別の親への誤委任を防ぐ。
  検証方法・証拠: fake gh による関係の変更・再実行テスト。
```

## 子タスク一覧と進捗の更新

親の「全体の完了条件」と「子タスクのチェックリスト」は別の項目。後者には**直属の**sub-issueを
open/closedとも1件1行で載せる。タスク名は子タイトルから親prefixと自身の識別名を除いた要約と一致させる。

```markdown
- [ ] #1217 親セッションの委任を整える
- [x] #1225 親子issueの書式を統一する
- [ ] #1230 旧案を実装する（取消）
```

未分割の場合は、一覧節の最初の非空行を `未分割（子issue未作成）` で始める。
この状態でも親Issue自身を実行できる。分割と委任は別判断で、[parent CLIの単独実行](../parent/README.md#issue自身を実行する)へ進める。
直後の句点「。」は任意で、同じ行の説明文や後続の分割候補の段落も記入できる。
この欄は子が0件の一覧として扱い、説明中のissue参照を子として抽出しない。
子issueのチェック行（書式が崩れた行を含む）との混在は拒否する。
分割後は未分割表記を外し、説明文や分割候補をスコープ・依存関係など適切な節へ移し、上のチェック行だけにする。

```markdown
未分割（子issue未作成）。起票段階では実装担当や仮の子番号を作らない。

分割候補は、記録契約、既存入口との接続、Git公開とREADME表示。
```

| 変更 | 担当親が行う更新 |
|---|---|
| 分割前 | 上記の未分割表記で記録する。未作成の番号・チェック行は置かない |
| 追加 | 子を作成し、正式なsub-issue関係を登録してから実番号の未チェック行を追加する |
| 分割 | 元の子に分割理由と引継ぎ先を記録し、新しい子を追加。置換なら元の子は取消として残す。元の子を親へ昇格するなら孫はその子の一覧にだけ載せる |
| 停止・依存保留・再開 | チェックは未完了のまま。子の詳細、親への通知コメント、親本文の進捗概要を更新する |
| 完了 | 子の検証・成果物を確認し、GitHubで `closed` / `completed` となった子だけ `[x]` にする。実装はPRのmergeでcloseする |
| 取消・重複 | 子に理由・残作業の行き先を書き、`closed` / `not_planned`（または `duplicate`）にする。関係と一覧を残し、未チェック＋`（取消）` にする。成功扱いにしない |
| 親の付替え | 旧親のチェック行を外して進捗概要に移動先を記録し、新親の一覧へ現在の状態で追加。取消の履歴として旧親にsub-issueを残さない |
| 再open | 取消suffix・完了チェックを外し、進捗と依存作業を見直す |

進捗概要には未着手・進行中・停止・依存保留・完了・取消の子番号と全体への影響を書く。
一覧のチェック数だけで親を完了させず、全体の完了条件と未決事項を担当親が確認する。

停止時、子の「作業状況・停止事項」に**理由・確認済みの事実、影響する依存作業、再開条件、次の対応**を残す。
親には次の短い通知コメントを残す（`#番号` はGitHub上で子へのリンクになる）。再開時も同じ形で通知する。

```text
停止: #1217 — 要件確認で保留。
全体への影響: #1218 は依存保留。独立した #1220 は継続。
次の対応: 担当親が判断を確認する。詳細・再開条件は #1217。

再開: #1217 — 判断を記録し実装を再開。
全体への影響: #1218 の保留を解除。#1220 は継続。
次の対応: #1217 の検証を進める。詳細は #1217。
```

子の担当が詳細を更新して担当親へ伝え、親コメントと全体進捗の更新は担当親が担う。
停止した子と影響する依存作業だけを保留し、独立した作業は継続する。

## 共通APIと下流の契約

Pythonのimport rootは `.agents/ops`、package名は `issues`。外部依存なし。

```python
from issues import (
    ChildTask,
    ParentRef,
    ProposalSource,
    child_title,
    parent_title,
    render_parent_body,
    render_child_body,
    check_consistency,
)

parent = ParentRef("開発ループ", 1216)
title = child_title(parent, "親子issueの書式を統一する")
body = render_child_body(parent.number, child_sections)
parent_body = render_parent_body(
    parent_sections,
    [
        ChildTask(1225, "親子issueの書式を統一する"),
        ChildTask(1217, "委任を整える"),
    ],
)
```

`parent_sections` / `child_sections` は上記必須見出しをキー、本文を値とする `Mapping[str, str]`。
`render_parent_body` へ子一覧を本文文字列で渡さず、`children` 引数の `ChildTask` で渡す。
`ChildTask.status` は `open` / `completed` / `cancelled`（既定 `open`）。
`parent_title(..., parent=ParentRef(...))` と `render_parent_body(..., parent_number=N)` で入れ子の親を生成する。
その場合は子の必須節もsectionsに含める。本文の各値に別の `##` 節を埋め込まず、別のキーにする。
生成時の形式違反は `ValueError`、`validate_body` はエラー文字列のtupleを返す。
`parse_title` / `parent_reference` / `parse_child_tasks` / `parse_sections` は検査・更新に再利用できる。
`PARENT_LABEL` / `CHILD_LABEL` / `STANDALONE_LABEL` をimportして付与し、他の軸のラベルを維持する。
親・子の `render` specでは役割ラベルを生成するため、任意の `labels` への3種の役割ラベルの混入を拒否する。

レポート経由でも、親は**レポートissueではなく開発用の親issue**。採用、Issue整理、人間合意、自律着手の根拠を
区別する。既存の手動親は [parent CLI](../parent/README.md#採用提案の受領)、未担当の採用提案は
[triage](../ops-report/README.md#採用提案の自動issue化) が扱う。書式・重複検出には以下を使う。

1. `render_child_body(..., source=ProposalSource(proposal_id, report_number, option))` を使う。
   提案IDは周期に依存しない不透明なASCII tokenとして受け取り、ID生成はレポート側に任せる。
2. メタデータは本文**1行目**の `<!-- ops-report-proposal: ID report=#N option=... -->`。
   `option` はpercent-encoding。親参照はその後に置く。既存の同形式メタデータも保持できる。
3. open/closedの候補本文を `index_proposals([(number, body), ...])` に渡す。
   戻り値は `(report_number, proposal_id) -> issue_number`。タイトルやprefixから検索・重複判定しない。
   同じキーの複数issue、不正・重複メタデータはエラー。`parse_proposal_source(body)` で現在の選択肢まで
   比較し、一致しない既存issueを再利用しない。タイトル改名・親付替えで重複起票しないことをテスト済み。

## 終了時改善の独立親

`issues.improvement.ImprovementSource(parent, key)` は `I-<元親>-<key>` の候補を表す。
`render` specの `improvement_source: {parent: N, key: KEY}` はrole=parent・正式親なしのときだけ使え、
本文1行目に `<!-- parent-improvement-source:v1 parent=N key=KEY -->` を置く。
`parse_source` / `index_sources` は全open/closedの本文から出典を照合し、重複・不正な出典を拒否する。
`contract` は出典マーカーを除く前文も比較する。通常文章の空白・改行を正規化し、fence内の文字列を保持する。
前文の保存キー `_preamble` は予約済みで、同名の見出しによる契約の上書きを拒否する。
`execution_terms(requirements)` は承認snapshotから実行合意の4欄を生成する。必須節と追加節の扱いは
[人間判断と実行](../parent/improvements.md#人間判断と実行)を参照する。
これは `ProposalSource` / `index_proposals`・`ai-proposed`・`parent-task:v1` とは独立した出典である。

`issues.creation.create_issue_once` は固定draft・送信前receipt・出典に一致する候補の照合を共有する。
呼出側は一致条件と永続化・排他を提供する。応答不明で一致先も未観測なら再POSTせず、明示的な再送時に照合する。
終了時改善のcallerは最初のPOSTに `issue:parent` / `needs-human` を含め、元親の子・依存に追加しない。
候補の正本・人間判断・既存issueへの集約・導入順は [parentの終了時改善](../parent/improvements.md)を参照する。
候補の受入と元親の全体受入を混ぜず、子テンプレートの改善欄は元親の正本コメントへの参照に使う。

## triageによる出典と復旧契約

`render_standalone_body(sections)` は子の具体的な必須節・Acceptanceを持ち、親参照を持たない単独作業の本文を生成する。
triageはrole=standaloneにだけ `ai-proposed` と `STANDALONE_LABEL` を付ける。
人間が単独テンプレートで起票する本文は自由形式で、親・子の必須節を要求しない。
親・子は従来の共通APIを使い、`ai-proposed` とそれぞれの役割ラベル、正式なsub-issue、親一覧を揃える。
レポートの `ops-report` ラベルと容器メタデータは作業issueへコピーしない。
reuseでは既存ラベルを保持し、standaloneを後付けしない。
保存済みpublication draftがある再開・復旧では、そのtitle・body・labelsを固定リクエストとして使う。
新ラベル導入前のdraftを再描画してラベルを差し替えず、作成済みissueも保存draftと照合する。
応答不明時は出典による実在確認を先に行い、未観測のPOSTを自動再送しない。入力・コード版の鮮度検査は維持する。

既存の `ProposalSource` / `index_proposals` は1提案→1つの**索引anchor**を維持する。
1提案を複数子へ分ける場合は親をanchorにする。複数提案を共通親へまとめる場合は各提案の子に別anchorを置く。
親には全提案のcontextを持たせる。1つのissueへの複数の提案markerや同一提案の複数issueへの登録は拒否する。
dispatcherは索引anchorに加え、contextの全bindingからも出典レポートを`source_reports`へ投影する。
anchorが別の正式親へ移っても、元の親・分割子の出典は対象閉包と分割入力に残る。索引の一意性は変えない。

`bind_task(body, plan=SHA256, node=KEY, bindings=[...], source=ProposalSource|None)` が生成する
`ops-report-task:v1` markerは本文先頭（anchorでは従来のproposal markerの直後）に置く。
`parse_task_metadata(body)` は位置・未知キー・canonical JSON・重複・digest・sourceとの一致を検査する。
各bindingは `source: {report_number, proposal_id, option}` と `proposal_sha256` を持ち、report番号・ID順に並べる。
内容digestは `ParsedProposal.snapshot()` のcanonical JSON SHA-256で、全選択肢・タイトル・目的・本文・受入条件を含む。
採用チェック・選択値・関連親・見送りは別途現在値を照合する。digestから人間合意を生成しない。

`plan_sha256` と `node` は、作成応答を失ったときに同じ計画のissueを全open/closed状態から復旧する識別子。
actor/sessionの所有権を表さない。既存作業をreuseするときは内容・タイトル・state・正式親・labelsを保持し、
出典markerと元提案の出典節だけを追加する。他の提案で索引済みのissueや取消された作業へ無断で統合しない。
計画の保存・書込前の再照合・中断復旧の手順は上記triage READMEを正本とする。

## CLIと整合検査

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m issues check --issue 1216
PYTHONPATH=.agents/ops .venv/bin/python -m issues check --repo Motoki0705/tennis-lab --issue 1225
PYTHONPATH=.agents/ops .venv/bin/python -m issues render --spec tmp/1225.spec.json > tmp/1225.draft.json
```

`render` は常にdry-runで、`title` / `body` / `labels` / `dry_run: true` のJSONを標準出力へ返すだけ。
`--dry-run` は明示してもよい。書き込み・自動修復・一括移行のコマンドは提供しない。
specは次の形。`role: parent` では `identifier` と任意の `children` を指定し、
`role: child` では `parent` が必須。入れ子の親は `role: parent` に `parent` も指定する。

```json
{
  "role": "child",
  "summary": "親子issueの書式を統一する",
  "parent": {"identifier": "開発ループ", "number": 1216},
  "labels": ["enhancement", "ci"],
  "sections": {
    "このタスクの目的": "親の目的に沿って書式の共通化を行う。",
    "具体的な変更内容・対象範囲": ".agents/ops/issues/ と起票テンプレート。",
    "対象外": "レビュー起動とtriageの改修。",
    "依存関係": "先行なし。下流は親セッションとtriage。",
    "作業状況・停止事項": "未着手。停止事項なし。",
    "Acceptance checklist": "- [ ] 矛盾を検出できる\n  文脈: 親を一意に特定するため。\n  下流への影響: 誤委任を防ぐ。\n  検証方法・証拠: fake gh で確認する。"
  }
}
```

親specの `children` は `[{"number":1225,"title":"親子issueの書式を統一する","status":"open"}]`。
未分割なら空配列。`source` は任意で `{"proposal_id":"ID","report_number":90,"option":null}`。
`labels` は作業種別・対象領域・起票元を指定する（親子ラベルは生成する）。未知のキーはエラー。

`check` は `gh repo view`（`--repo` 省略時）とGETだけを使い、
[GitHub REST sub-issues API](https://docs.github.com/en/rest/issues/sub-issues) の `.../parent` と
`.../sub_issues`（100件ずつ全ページ）で直属の関係を取得する。指定したissueから祖先・兄弟・子孫を辿り、
closed/取消を含む一式のラベル、prefix、本文参照、必須項目、子一覧の番号・要約・状態、循環を検査する。
正式な関係を正本とし、本文などを元に勝手に関係を追加しない。別repoのsub-issueは明示的なエラーとする。

終了値は整合=0、食い違い=1、取得失敗・不正入力=2。取得失敗時は部分的な成功結果を出さない。
既に取得できたissueの `/parent` のHTTP 404 のうちGitHubが親の不在に返す `No parent issue found` だけを「親なし」とし、403・5xxや他の404（issue自体が無い `Not Found` を含む）は失敗。
closedの理由が不明な子を完了と推測しない。編集中に跨った取得は不整合になり得るため、編集を終えて再検査する。
自然言語のスコープの矛盾・合意の正当性・進捗概要と通知コメントの内容は担当親が確認する。
`label.standalone` はstandaloneと親子ラベル・正式な親子関係・子一覧節・親参照・子タイトルprefixの矛盾を示す。
役割ラベルのない既存issueに新ラベルを要求する検査は行わない。

### GitHub GETの接続断と再試行

`issues.cli.GitHub.get()` は、ghが失敗終了しHTTP応答statusが得られないGETだけを合計3回まで試行し、
2回目の前に1秒、3回目の前に2秒待つ。各試行のsubprocess timeoutは60秒。
`error connecting` 等を伴うstatus欠落と、HTTP statusを受信する前のtimeoutが対象。
通信原因をDNS・TLS・回線などのいずれかと断定するものではない。

HTTP statusを受信したら再試行しない。認証・権限の401/403、404、rate limitの403/429、5xxは
`GitHubHTTPError` の実statusを保持して失敗させる。rate limitの解除待ちや自動再送は行わない。
HTTP statusがある応答の本文不正・欠落も即失敗する。上記の文書化された「親なし」404だけは従来どおり扱う。
ghが正常終了したのにHTTP statusがない応答は、接続断とみなさず不正応答として即失敗する。
実行ファイルの欠落等のローカルエラー、`gh repo view`、POST/PATCH/PUT/DELETE、GraphQLのPOSTは再試行しない。

上限に達したら `GitHubTransportError` にGET endpoint、`attempts`、`last_error_kind` と最後の詳細を保持する。
エラー種別は `missing_http_response_status` または `timeout_without_http_response`。
再試行後にHTTPエラーへ変わった場合も、HTTPエラーに累積試行回数と最終statusを残す。
空配列や成功には置き換えない。親・dispatcher等の共有reader利用先にも同じGET契約が適用される。
`parent.github.GitHub.rows()` の最大3回の取り直しはページ重複の整合対策であり、接続断の再試行とは別である。

## 新規起票と既存issueの段階移行

**新規は起票時、既存は着手・再開・親付替えなど今後作業対象になる時**に、起票者と担当親が適用する。
対象外のOpen全件・Closedを含む過去全件の一括改名・ラベル付与・本文移行は行わない。
#1284で合意した一度だけのopen issueの誤ラベル解除は例外とし、上記 `strip-auto-labels` で行う。
standaloneの初回移行は [#1878](https://github.com/Motoki0705/tennis-lab/issues/1878) の指定11件だけを再確認し、
付与済み・対象外・競合/未解決と根拠を記録する。付与数だけで完了とせず、競合を未解決のまま初回移行完了としない。
途中失敗しても新issueを作り直さず、実番号・現在の関係を再取得して不足分を揃える。

### 単独から親・子への明示移行

1. 対象issueと編入先の本文・タイトル・コメント・全ラベル・担当・open/closedと理由・正式な親/子を再取得して退避する。
   API失敗を「関係なし」へ読み替えない。保存済みtriage計画・journalや進行中処理がラベル一致を前提にしていれば、
   その処理と照合・復旧してから移行する。公開途中の固定draftと異なるラベルを外付けしない。
2. 昇格では親の識別名・必須節・子一覧を整え、子0件なら未分割表記を使う。
   編入では正式親を確認し、子のprefix付きタイトル・冒頭の親参照・必須節を整える。
   共通APIまたは `issues render` のdraftを元の内容と比較し、合意・Acceptance・履歴の参照や追加節を保持する。
3. UTF-8の本文ファイルを用意し、役割追加とstandalone除去を本文・タイトルと同じ1回の編集で行う。
   下の変数には再取得して確認した番号・タイトルを指定する。他のラベル・担当・状態・履歴は保つ。

   昇格（単独→親）:

   ```bash
   gh issue edit "$task_issue" --repo Motoki0705/tennis-lab \
     --title "$parent_title" --body-file tmp/task.body.md \
     --add-label issue:parent --remove-label issue:standalone
   ```

   編入（単独→既存親の子）:

   ```bash
   gh issue edit "$task_issue" --repo Motoki0705/tennis-lab \
     --title "$child_title" --body-file tmp/task.body.md \
     --add-label issue:child --remove-label issue:standalone
   ```

4. 昇格後に子を作る場合は親CLIを使う。standaloneが残った親の分割は拒否する。
   編入では正式な関係を再確認し、未登録の場合だけ次のPOSTを行う。`task_rest_id` はGETで確認した子のRESTのidで、
   issue番号ではない。既に正しい関係なら再追加しない。予期しない親があれば自動上書きせず担当親間で解消する。

   ```bash
   gh api --method POST "repos/Motoki0705/tennis-lab/issues/$parent_issue/sub_issues" \
     -F sub_issue_id="$task_rest_id"
   ```

5. 正式関係から親一覧・進捗を更新し、`issues check --issue "$task_issue"`（編入先も同様）が0になるまで照合する。
   変更前後の状態・ラベル・関係と理由をPR/issueへ記録する。分類だけを理由に作業を起動・再open・closeしない。
   途中失敗では現在の本文・ラベル・正式関係・一覧を再取得し、不足分だけを揃えて再検査する。
   ラベルが既に揃っていれば編集を繰り返さず、既存の関係も再POSTしない。親CLIの子作成復旧も、
   他のラベルを保持してstandaloneだけを除き、ラベルが一致していればPATCHしない。

### 親子規約導入時の移行

#1225で対象を限定した #1216, #1217〜#1223, #1225 の移行は次の手順を使う。

1. 管理者が `issue:parent` / `issue:child` ラベルを作成する（既存なら内容を確認）。
2. 担当親が対象の本文・コメント・ラベル・open/closedと理由・sub-issue関係をGETして退避し、
   #1216の識別名を「開発ループ」に決める。採否・要件合意コメントと本文の不一致は最新の合意で解消する。
3. 各issueの既存文章を必須節へ対応付けたspecをworktreeの `tmp/` に作る。合意内容・採否・
   Acceptanceのチェック状態・依存関係・進捗・旧管理メタデータを捨てない。追加の節も保持する。
   `render` でdraftを生成し、元本文と比較する。proposalメタデータがある場合はsourceへ移し1行目を維持する。
   その他の旧管理コメントは本文に戻してから `validate_body` でも確認する。
4. 担当親がdraftのbodyをUTF-8ファイルに取り出し、明示した番号にだけ `gh issue edit --body-file` と
   `--title` / `--add-label` を適用する。既存の作業種別・対象領域・起票元ラベルとissueの状態は変えない。
   例: `gh issue edit 1225 --title '[開発ループ #1216] 親子issueの書式を統一する' --add-label issue:child --body-file tmp/1225.body.md`。
5. 正式な関係が未登録なら、確認した親にsub-issues APIのPOSTを行う。`sub_issue_id` は子の**RESTのid**で、
   issue番号ではない。既に正しい関係なら再追加しない。親付替えが必要なら旧親から外して新親へ登録し、
   両親の一覧・進捗と子のprefix・本文を更新する。予期しない親への所属は自動上書きせず担当親間で解消する。
6. 各親の一覧を実際の状態に合わせ、`check --issue 1216` が0になるまで具体的な食い違いを修正する。
   番号・関係・採否・合意・完了状態の移行前後比較と、親への停止/再開通知・進捗を確認して移行記録を残す。
   GitHub上での書き込みと実検証は親が実施する。ホスト設定の変更は不要。

テスト: `.venv/bin/python -m pytest tests/e2e/agents_ops/test_issue_format.py -n 4 -q`。
