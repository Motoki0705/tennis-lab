# 親子issueの規約と共通ツール

タイトル・ラベル・本文・親子関係・進捗記録の正本。手動起票、親セッションによる分割（#1217）、
レポート提案からの起票（#1223）はこの規約と共通APIを使う。親セッションの割当・要件合意・
レビュー・マージの実装はここでは扱わない。

## 役割とテンプレートの選択

| ラベル | 意味・付与条件 |
|---|---|
| `issue:parent` | 全体の目的・合意・進捗を管理する親。分割前にも付ける |
| `issue:child` | GitHubの正式なsub-issueとして直属の親を持ち、具体的な1タスクを担当する子 |

両方を兼ねるissueには両ラベルを付ける。別軸のラベルは親子区分の代わりにならない。
ラベルから親番号を推測しない。

| 利用場面 | テンプレート |
|---|---|
| 全体の構想・複数タスクの統括 | [親issue](../../../.github/ISSUE_TEMPLATE/06_parent.md) |
| 親の合意範囲から分割した1タスク | [子issue](../../../.github/ISSUE_TEMPLATE/07_child.md) |
| 親を持たない単独タスク | 既存の[バグ](../../../.github/ISSUE_TEMPLATE/01_bug.md)・[機能](../../../.github/ISSUE_TEMPLATE/02_feature.md)・[調査](../../../.github/ISSUE_TEMPLATE/03_research.md)。親子ラベルは付けない |
| 研究ループ | [研究テーマ](../../../.github/ISSUE_TEMPLATE/04_research_theme.md)は親、[研究サイクル](../../../.github/ISSUE_TEMPLATE/05_research_cycle.md)は子。共通項目に研究固有の項目を加える |

単独タスクを分割するときは親へ移行する。単独タスクを既存の親へ編入するときは子へ移行する。
研究の指標・評価条件・合意・知見の置き場やclose判断は[research-loop](../../skills/research-loop/SKILL.md)に従う。

## ラベル

名前・色・説明と軸ごとの一覧の正本は [.github/labels.yml](../../../.github/labels.yml)。
意味と付与主体はここで定義する。ラベルの作成・更新・削除は管理者が下記の同期CLIで行う。

| 軸（マニフェストのキー） | 意味 | 付与主体 |
|---|---|---|
| 階層（`hierarchy`） | 親・子の役割（上表） | 起票者・担当親・親CLI。`issues check` で正式な関係と照合する |
| 作業種別（`work-type`） | バグ修正・機能改善・調査研究・文書・追補 | テンプレート、起票者・親CLIの明示指定、issue-labelerのタイトル先頭prefix |
| 起票元（`origin`） | `ai-proposed` は運用レポートの採用提案からの起票、`agent:*` は起票agent | 前者はレポートtriage、後者は起票agent。通常のAI作業・親CLIの子起票には `ai-proposed` を付けない |
| 対象領域（`area`） | 対象モジュール・CI・テスト・依存・外部コード・Python・知見 | issueは起票者・親CLIの明示指定だけ。PRは `.github/labeler.yml` が変更ファイルから付け、Dependabotは設定で指定する |
| 状態（`status`） | `needs-human` は人間の承認待ち、`ops-report` は運用レポートの容器、`rewrite:*` は[本文書き直しの確定状態](#親issueの書き直し状態) | 承認を必要とする仕組み・担当者、レポート生成CLI、書き直しworkflow（skipは人間） |

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

## 親issueの書き直し状態

判定の正本は `issues.rewrite_state`。ローカルregistryやworkflowの終了状態から本文の確定を推測しない。
有効化の正本はrepository variable `ISSUE_REWRITE_ENABLED_SINCE`（UTCの `YYYY-MM-DDTHH:MM:SSZ`）。
未設定のHTTP 404だけを無効とし、403・5xx・通信失敗・不正な応答や日時は例外にする。
設定とGitHub上のラベル作成は管理者が別途行う。

有効かつ現在 `issue:parent` を持ち、GitHubの `created_at` が有効化時刻**以降**のissueだけが対象。
それ以外は `not_target`。対象で確定ラベルがなければ `pending`、あれば以下の状態になる。
確定ラベルが2種類以上付いていたら、非対象のissueでも例外にする。

| 公開定数 | ラベル | 状態・意味 |
|---|---|---|
| `REWRITE_DONE_LABEL` | `rewrite:done` | `done`: 書き直し完了 |
| `REWRITE_FAILED_LABEL` | `rewrite:failed` | `failed`: 書き直しに失敗し、原文で確定 |
| `REWRITE_SKIPPED_LABEL` | `rewrite:skipped` | `skipped`: 人間が書き直しを省略 |

### Python API（workflowとの共有契約）

import rootは `.agents/ops`。workflowも次の公開名をそのまま使う。

```python
from issues.cli import GitHub
from issues.rewrite_state import (
    REWRITE_DONE_LABEL,
    REWRITE_FAILED_LABEL,
    REWRITE_SKIPPED_LABEL,
    REWRITE_FINAL_LABELS,
    REWRITE_ENABLED_SINCE_VARIABLE,
    read_enabled_since,
    rewrite_state,
)

gh = GitHub("owner/repo")
enabled_since = read_enabled_since(gh)
issue = gh.get("repos/owner/repo/issues/123")
state = rewrite_state(
    [label["name"] for label in issue["labels"]],
    issue["created_at"],
    enabled_since,
)
```

- `read_enabled_since(gh) -> str | None`: `repo` と `get(endpoint) -> object` を持つ読み取りclientを受け取る。
  共通 `issues.cli.GitHub`（`parent.github.GitHub` も可）が `gh api --include --method GET` で
  `repos/{repo}/actions/variables/ISSUE_REWRITE_ENABLED_SINCE` を取得する。
  `GitHubHTTPError.status == 404` のときだけ `None`、設定済みなら検証したUTC文字列を返す。
- `rewrite_state(labels, created_at, enabled_since) -> RewriteState`: `labels` はラベル**名の文字列**のiterable。
  RESTのラベルobjectそのものは渡さない。日時はUTC秒形式の文字列、無効の場合だけ `enabled_since=None`。
  不正な日時・確定ラベルの競合は `ValueError`。戻り値の型 `RewriteState` は上記5状態の `Literal`。
- `REWRITE_FINAL_LABELS` は3つの確定ラベルの `frozenset[str]`、
  `REWRITE_ENABLED_SINCE_VARIABLE` はrepository variableの名前。
  定数・型・`read_enabled_since` は `issues` packageからも公開する。
  `from issues import rewrite_state` は関数ではなく同名の**module**を返す。

親CLIのgateと待機は [parent README](../parent/README.md#書き直しの確定待ち)、
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

| 変更 | 担当親が行う更新 |
|---|---|
| 分割前 | 一覧節に `未分割（子issue未作成）` と書く。未作成の番号・チェック行は置かない |
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
`PARENT_LABEL` / `CHILD_LABEL` をimportして付与し、作業種別・対象領域・起票元ラベルを維持する。

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

## triageによる出典と復旧契約

`render_standalone_body(sections)` は子の具体的な必須節・Acceptanceを持ち、親参照・親子ラベルを持たない
単独作業を生成する。親・子は従来の共通APIを使い、`ai-proposed` と役割ラベル、正式なsub-issue、親一覧を揃える。
レポートの `ops-report` ラベルと容器メタデータは作業issueへコピーしない。

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

## 新規起票と既存issueの段階移行

**新規は起票時、既存は着手・再開・親付替えなど今後作業対象になる時**に、起票者と担当親が適用する。
対象外のOpen全件・Closedを含む過去全件の一括改名・ラベル付与・本文移行は行わない。
#1284で合意した一度だけのopen issueの誤ラベル解除は例外とし、上記 `strip-auto-labels` で行う。
途中失敗しても新issueを作り直さず、実番号・現在の関係を再取得して不足分を揃える。

今回の導入後は担当親が明示的に #1216, #1217〜#1223, #1225 だけを次の順で移行する。

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
