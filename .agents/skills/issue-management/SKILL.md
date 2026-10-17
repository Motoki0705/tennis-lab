---
name: issue-management
description: Manage tennis-lab issues and progress in GitHub Project 3, including parent/sub-issue decomposition, body templates, triage, status updates, and closure.
---

# Issueと進捗の管理

管理先は [prj-tennis-lab（Motoki0705 / Project #3）](https://github.com/users/Motoki0705/projects/3)。
このskillはIssue管理の共通手順の正本。本文のひな形は `.github/ISSUE_TEMPLATE/` を正本とする。
新しいProjectや、本文だけの別の進捗台帳を作らない。

## 情報の置き場所

| 情報 | 正本 |
|---|---|
| 状態・優先度・担当・日程 | Projectの `Status` / `Priority` / `Assignees` / 日付フィールド |
| 親子関係・子の集計 | GitHub標準のSub-issues、Projectの `Parent issue` / `Sub-issues progress` |
| 目的・対象範囲・完了条件・現在の到達点・次の一手 | Issue本文 |
| 判断の経緯・実行ログへの参照 | Issueコメント。最新の結論は本文へ要約する |
| 実装・検証 | PR。実験の数値・考察は [knowledge-control](../knowledge-control/SKILL.md) |

最新の進捗要約は本文冒頭に置く。長い仕様・実行ログはREADMEや成果物への参照に分ける。
親本文に子のStatusや完了率を手で複製しない。Sub-issues progressは件数の集計であり、
工数・品質・親の受入条件の達成率ではない。中止・重複のcloseも成功と混同しない。

## 起票・分割

1. Project、open/closed Issue、関連PRを検索し、同じ目的のIssueがあれば更新する。
   不明点が成果・範囲・完了条件を変える場合は、起票前に人間と具体化する。
2. **親は達成したい成果のまとまり、子は単独で検証・完了判定できる作業**とする。
   原則2階層。1つの作業で完結するものは単独Issueでよく、分類のためだけの親を増やさない。
   子のさらに分割が必要なら、まず粒度を見直し、独立した完了条件が必要な場合だけ深くする。
3. 親の完了条件に `AC1` などのIDを付け、各子がどれを満たすかと成果物・責務境界を対応付ける。
   共通の仕様は親に置き、子には参照とその作業に固有の条件を書く。未確定の演出や指標を
   分割しただけで合意済みにしない。
4. 別Issueにするのは、成果物・完了条件・依存関係を独立して追跡する必要がある作業。
   同じ成果の細かな手順や軽微な修正はチェック項目にする。実験のrunごとには作らず、
   [research-loop](../research-loop/SKILL.md) の1サイクルを単位にする。
5. 子を作ったら**標準のSub-issues関係を登録**し、親と子の両方をProjectへ登録する。
   本文の `親Issue: #...` やリンクのチェックリストだけでは完了しない。
   同じ子の親は1つ。複数の親に役立つ共通作業は主たる成果に所属させ、他からは依存として参照する。
6. 分割関係と実行順の依存関係を区別する。先行Issueの完了が本当に必要ならGitHubの
   `blocked by` 関係を付け、本文に必要な成果物と解除条件を書く。単なる関連リンクでは代用しない。

AIは作業中に見つけたバグ・負債・派生課題を、この粒度で起票してよい。
AI作成Issueには `ai-proposed` を付ける。提案の起票は着手の合意を意味せず、初期状態は
`Backlog` とする。承認済みの作業なら、その根拠と実態に合わせて更新する。

## 本文フォーマットの選択

| 用途 | テンプレート |
|---|---|
| 複数の実装・調査をまとめる親 | [00_parent.md](../../../.github/ISSUE_TEMPLATE/00_parent.md) |
| 通常の子・単独の実装タスク | [02_feature.md](../../../.github/ISSUE_TEMPLATE/02_feature.md) |
| 不具合の子・単独タスク | [01_bug.md](../../../.github/ISSUE_TEMPLATE/01_bug.md) |
| 調査の子・単独タスク | [03_research.md](../../../.github/ISSUE_TEMPLATE/03_research.md) |
| 指標・停止条件を合意する研究テーマ | [04_research_theme.md](../../../.github/ISSUE_TEMPLATE/04_research_theme.md) |
| 研究テーマの子（1サイクル） | [05_research_cycle.md](../../../.github/ISSUE_TEMPLATE/05_research_cycle.md) |

`gh issue create --body-file` に渡す本文には、テンプレートのYAML frontmatterを含めない。
空欄や説明コメントを埋め、対象外なら理由を記す。通常の子は親・親の完了条件との対応・
担当範囲・成果物・依存と解除条件・検証できる完了条件を持つ。単独Issueは親なしと明記する。
タイトルは成果・変更内容を表し、親番号は本文の追跡情報に置く。
研究ではAC番号の代わりにテーマの指標・停止条件との対応を示す。

## 状態と更新のタイミング

Project既存の5つのStatusを使う。状態名をラベルや本文の独自ステータスで二重管理しない。

| Status | 判断基準 |
|---|---|
| `Backlog` | 提案・未着手。方針や受入条件が未確定、先行作業待ち、優先順位未決も含む |
| `Ready` | 次に着手する対象として選定済みで、範囲・完了条件・必要な入力が揃い、依存が解消している |
| `In progress` | 調査・実装・実験を開始した。部分PRやDraft PRがあっても残作業があればここ |
| `In review` | そのIssue全体の成果・検証が揃い、レビュー・merge・人間の最終判定を待つ |
| `Done` | Issueをcloseした。成功か中止かはclose reasonと結論で区別する |

- **作成時**: Project登録・親子関係・Statusを再取得して確認する。自動登録だけに頼らない。
- **着手時**: Issueの目的・依存・関連PRを読み、担当と `In progress` を更新する。
  エージェント名を架空のGitHub assigneeとして扱わず、実行主体・worktreeなどは必要に応じて本文に残す。
- **節目・引継ぎ時**: 本文冒頭の「進捗要約」を更新する。確認日、到達点と証拠、次の一手、阻害要因と
  解除条件、関連PRを短く書く。細かな経緯はコメントに置く。
- **止まった時**: 開始後は現在の工程のStatusを維持し、阻害要因と再開条件を明記する。
  人間の判断が必要なら `needs-human` を付け、解消したら外す。待ち時間だけで完了へ進めない。
- **PR作成・更新時**: Issueのどの条件を満たすかをPRに書く。Issue全体を完了させるPRのみ
  `Closes #...` を使い、部分実装は `Refs #...` と残条件を記す。PRの存在だけで `In review` にしない。
- **merge・結果確定時**: 完了条件と証拠を確認し、未達ならIssueを開いたまま残作業を明記する。
  対応する親の到達点・次の一手・Statusも更新する。

親は子の着手で `In progress` に進め、全体の成果と検証が揃ったら `In review` にする。
子がすべて閉じても親の完了条件を確認する。進行中・未確認の子がある親を閉じない。
子が1件review中であることだけでは親を `In review` にしない。

## 完了・中止

- 実装Issueは原則、完了条件を満たすPRのmergeによる `Closes #...` でcloseする。
  複数PRなら最後のPRだけに指定する。すでにmerge済みなら、証拠を結論に残してcloseする。
- 実装を伴わない調査は、問いへの回答・証拠・残った不確実性を残してcloseする。
  研究サイクルとテーマの判定は [research-loop](../research-loop/SKILL.md) に従い、テーマのcloseは人間が行う。
- 親は子の統合結果と各完了条件の証拠を確認して閉じる。未達の条件を子のclose件数で代用しない。
- 重複・中止は理由と統合先を残し、`not planned` で閉じる。範囲の取り消しが未決なら判断を求める。
- **Issueのclose後にProjectのDoneを確認する。** ProjectのAuto-close issueが有効だと、
  Doneへの移動がIssueのcloseを引き起こすため、Doneを先に付けて完了扱いしない。

## Project操作・棚卸し

フィールド、推奨ビュー、CLI/API、既存Issueの移行手順は
[project-operations.md](references/project-operations.md) を操作時に読む。
週次や進捗確認時は、全体→親ごとの子→判断待ち・古い情報の順に確認する。
登録漏れ、Status未設定、親子リンク漏れ、レビュー待ちの放置、閉じたIssueとの不整合を点検し、
件数を減らすためだけのcloseや根拠のない状態変更はしない。

[週次レポート](../../ops/weekly-report/README.md) の自動生成は別の実装経路を持つ。
レポートは提案の出所であり、各提案の成果をまとめる親とは限らない。自動生成された提案を
triageするときもこのskillで所属とProjectを確認する。機械照合用マーカーは保持する。
