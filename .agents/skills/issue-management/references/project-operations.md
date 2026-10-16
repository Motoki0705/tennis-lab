# Project #3の操作と棚卸し

管理先と運用判断は [SKILL.md](../SKILL.md) を参照する。ここは操作時だけ読む。
取得権限・書込権限が足りない場合は、未反映の操作と必要な権限を報告する。
新しいProjectや本文リンクのみの管理へ黙って切り替えない。

## フィールドとビュー

| フィールド | 使い方 |
|---|---|
| `Status` | skillの5状態。作成・着手・レビュー・完了の節目に更新 |
| `Priority` | `P0`: 緊急の阻害要因、`P1`: 現在優先する成果、`P2`: その後。未判断なら空欄のままtriage対象にする |
| `Parent issue` / `Sub-issues progress` | 標準Sub-issuesから表示。手入力する独自フィールドは作らない |
| `Assignees` | 実際の担当者が決まったら設定 |
| `Start date` / `Target date` | 実際の開始日・合意した期限だけ記入。期限を推測しない |
| `Size` / `Estimate` | 必要なときだけ利用。埋めるための見積りはしない |
| `Linked pull requests` | PRの関連を確認。部分PRもあるのでリンクの存在を完了の根拠にしない |

同じProject内に、用途別の保存ビューを用意する。これは設定案であり、skillを追加しただけでは
既存ビューは変わらない。既存ビューと同じ用途なら重複作成せず再利用する。

| ビュー | フィルタ | 表示・用途 |
|---|---|---|
| 全体 | `repo:Motoki0705/tennis-lab is:issue is:open no:parent-issue` | Table。親と単独IssueのStatus、Priority、Sub-issues progress、次の期限を見る |
| 実行 | `repo:Motoki0705/tennis-lab is:issue is:open` | BoardをStatusで分け、Parent issueを表示・グループ化。親ごとに掘り下げる |
| レビュー待ち | `repo:Motoki0705/tennis-lab is:issue is:open status:"In review"` | 全受入条件が確認待ちのIssue |
| 判断待ち | `repo:Motoki0705/tennis-lab is:issue is:open label:needs-human` | 人間に必要な判断と解除条件を見る |
| 未整理 | `repo:Motoki0705/tennis-lab is:issue is:open no:priority` | 優先度未判断。`no:status` は別フィルタで確認する |

特定の親を確認するときは `parent-issue:Motoki0705/tennis-lab#<番号>` を使う。
初期表示はopen Issueに絞り、close済みはフィルタを外して確認する。既存のPRを削除する必要はない。
`has:` / `no:` を含む [フィルタ仕様](https://docs.github.com/en/issues/planning-and-tracking-with-projects/customizing-views-in-your-project/filtering-projects) と
[親子フィールドの仕様](https://docs.github.com/en/issues/planning-and-tracking-with-projects/understanding-fields/about-parent-issue-and-sub-issue-progress-fields) を参照。

## 取得

```bash
gh project view 3 --owner Motoki0705 --format json
gh project field-list 3 --owner Motoki0705 --format json
gh project item-list 3 --owner Motoki0705 --limit 1000 --format json
gh issue list --repo Motoki0705/tennis-lab --state open --limit 1000 \
  --json number,title,url,body,labels,assignees,updatedAt
```

件数上限で切れた一覧を全件として扱わない。Projectは `totalCount` と取得件数を照合し、
上限に達するIssue一覧や大きなProjectはGraphQLのcursorで全ページ取得する。
field ID・option ID・item IDは取得結果から名前・Issue URLを照合して解決し、固定値を保存しない。
Projectのitem ID、Issueのnode ID、RESTの数値ID、Issue番号は別物である。

## 作成と関連付け

以下の書き込み例は、対象Issueの作成・管理が依頼の範囲にあるときだけ使う。
調査や運用案の作成だけを頼まれた場合は実行しない。番号やファイル名は対象に置き換える。

```bash
# 通常の子の例。先に親と既存Issueを確認し、02_feature.mdを埋めた本文を用意する。
gh issue create --repo Motoki0705/tennis-lab \
  --title '[機能] 検証できる成果を表すタイトル' \
  --label enhancement --label ai-proposed --body-file /tmp/child-issue.md

# 親と子の両方に実行。戻り値のidはProject item ID。
gh project item-add 3 --owner Motoki0705 --url "$ISSUE_URL" --format json

# 子の現在の親を確認。既に同じ親ならPOSTは不要。別の親なら意図を確認し、強制移動しない。
gh api "repos/Motoki0705/tennis-lab/issues/$CHILD_NUMBER/parent"

# CHILD_DATABASE_IDはIssue番号でなくRESTの.id。
CHILD_DATABASE_ID=$(gh api "repos/Motoki0705/tennis-lab/issues/$CHILD_NUMBER" --jq .id)
gh api --method POST "repos/Motoki0705/tennis-lab/issues/$PARENT_NUMBER/sub_issues" \
  -F sub_issue_id="$CHILD_DATABASE_ID"
gh api --paginate "repos/Motoki0705/tennis-lab/issues/$PARENT_NUMBER/sub_issues" \
  --jq '.[].number'
```

親なしの子の `/parent` は404を返すことがある。Issue本体へのアクセス成功を先に確認し、
認可・通信の失敗を親なしとして処理しない。API仕様は
[Sub-issues REST API](https://docs.github.com/en/rest/issues/sub-issues) を参照。
依存関係はIssueの `Relationships` → `Blocked by` で登録し、親子関係と別に検証する。

Issue作成後にProject登録や親子登録が失敗したら、作成済みURLと未反映の操作を残す。
再実行時はそのIssueを再取得し、不足した操作だけを行う。重複Issueを作り直さない。
本文の更新前には最新本文を読み直し、他の編集・履歴・機械照合用マーカーを保持する。

## 状態の変更と確認

取得済みのProject ID、item ID、Status field ID、目的のStatus option IDを使う。
以下は1フィールドの更新例であり、IDが未解決のまま実行しない。

```bash
gh project item-edit --project-id "$PROJECT_ID" --id "$ITEM_ID" \
  --field-id "$STATUS_FIELD_ID" --single-select-option-id "$STATUS_OPTION_ID"
gh issue view "$ISSUE_NUMBER" --repo Motoki0705/tennis-lab --json state,stateReason,body
gh project item-list 3 --owner Motoki0705 --limit 1000 --format json
```

更新後に対象Issueの状態・Projectの値を再取得して一致を確かめる。
ProjectのWorkflowsも確認する。自動登録、PRリンク時の状態変更、close時のDoneなどは補助であり、
有効なだけで設定内容・対象・成功までは保証されない。特にAuto-close issueが有効なProjectでは、
StatusをDoneへ動かす前に [完了・中止の手順](../SKILL.md#完了中止) を満たす。
自動化の設定は、意図したものか確認してから変更し、部分PRでの誤closeを避ける。

## 既存Issueの移行

1. 全open Issue、既存の親子関係、関連PR、Projectの項目・Status・フィールド・Workflowsを取得する。
2. 「現状 / 変更案 / 根拠 / 未確認」の一覧を作る。本文に明記された親子、PRの受入条件との対応、
   merge状態などを根拠にする。Issue数・タイトル・更新日だけから完了を推測しない。
3. 本文だけで関連付いている子は標準Sub-issuesへ接続し、親子ともProjectに存在することを確認する。
   元本文の合意・完了条件・履歴は保つ。重複した子一覧のStatusは、標準表示への案内に置き換える。
4. 実際の着手・残作業・検証・PRの状態に従いStatusを修正し、親の要約も更新する。
   優先度・中止・所属が決められなければ、未確認の判断として人間に返す。
5. 保存ビューを整え、登録漏れ・親子漏れ・Status未設定・closeとの不整合を再取得して確認する。

週次レポートの生成コードは現状、本文リンクによる提案Issueの作成までを行う。
その自動化にProject操作や標準Sub-issues登録を追加するのは別の実装作業であり、skillの追加だけで
対応済みとみなさない。手動triageで所属・登録を補い、提案IDと重複防止マーカーを保持する。
