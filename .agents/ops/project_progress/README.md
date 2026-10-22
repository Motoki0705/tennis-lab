# 親子issueのProject表示

既存の [prj-tennis-lab](https://github.com/users/Motoki0705/projects/3) を使い、担当親が進捗を表示する。
要件・受入条件・停止詳細の正本はissue、担当セッションの正本は [parent registry](../parent/README.md#責務と状態の正本)。
Projectは表示先であり、別の全件台帳・ダミーissue・Project・担当stateは作らない。

## 項目とビュー

既存Status（Backlog / Ready / In progress / In review / Done）、Priority、Sizeと既存ビューは維持する。
追加項目の定義・選択肢の順序は [schema.py](schema.py) が正本。

| 追加項目 | 内容 |
|---|---|
| 段階 | 未着手 → 着手可能 → 作業中 → レビュー中 → 受入待ち → 完了 |
| 待ち理由 | テキスト。ユーザー判断、依存、CI失敗、権限不足などの理由・再開条件への参照。空文字指定で解除 |
| 次の作業 | 親が指定する次の対応。親行には集計を自動追記 |
| 親の受入状態 | 未確認 / 受入待ち / 受入済み。親だけに設定し、子のcloseから推測しない |

Assignees、Parent issue、Sub-issues progress、Linked pull requests、Priorityは既存項目を表示する。
Assigneesが空ならGitHub上の担当は未設定であり、ローカル親session IDから補完しない。
issue担当の指定は既存のGitHub運用で行う。このツールはissueのassigneeや正式な親子関係を変更しない。

| 追加ビュー | レイアウト・フィルタ | 用途 |
|---|---|---|
| 研究開発全体の一覧 | Table / `is:issue has:"親の受入状態"` | 正式タイトル、標準進捗、親の受入状態、次の対応 |
| 子タスクのカンバン | Board / `is:issue label:issue:child -reason:"not planned"` | 段階を列、Parent issueを横のグループにする |
| 判断/ブロック一覧 | Table / `is:issue has:"待ち理由"` | 段階を維持したまま待ち理由・次の対応を確認 |

カンバンに `parent-issue:"Motoki0705/tennis-lab#1204"` を追加すると、その親の直属子だけに絞れる。
共通ビューのフィルタには特定の親番号を固定しない。新しい親は初期登録から同じ入口を使う。

## 導入・更新の入口

書込みは担当するactiveな親セッション自身が [parent CLI](../parent/README.md) から明示実行する。
`status` / `delegate` / `sync` / `stop` / `resume` / merge gateには自動呼出しを追加していない。
共通の `--repo` / `--parent` はサブコマンドの前。Projectは `--owner` / `--project` で既存の参照先を指定できる。
既定は上記ユーザーProject。認証中のghにProject読取・更新権限が必要。

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1204 project-setup --dry-run
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1204 project-setup --apply
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1204 project-sync --event initial --spec tmp/project-initial.json --dry-run
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1204 project-sync --event initial --spec tmp/project-initial.json --apply
```

両コマンドとも省略時はdry-run。`--apply`だけがGitHubに書き込む。
setupは追加項目と3ビューの作成、フィルタ・表示項目の設定まで行い、`manual_steps`に残るUI作業を返す。
syncは `{ "schema_version": 1, "items": [...] }` を受け取る。初期登録は正式な親と全子孫を1件ずつ指定する。
その後の節目は変更対象だけを指定できる。未登録の子が増えた場合、その子の明示値も必要。
入力にはrepo・ローカルsession ID・worktreeパスを入れない。未知キー・重複issue・対象の正式な子孫でないissueは拒否する。

着手時の例（初期登録済みの子）:

```json
{
  "schema_version": 1,
  "items": [{
    "issue": 1207,
    "stage": "作業中",
    "wait_reason": "",
    "next_action": "合意済みの素材で実装と検証を進める"
  }]
}
```

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m parent --parent 1204 project-sync --event start --spec tmp/project-start.json --apply
```

| event | 更新責任者とタイミング・検査 |
|---|---|
| initial | 担当親が導入・新しい親への展開時に、正式な親子すべての値を指定して初期登録 |
| refresh | 担当親が着手可否・次の対応の変更や取消を確認後に明示更新 |
| start | 子を着手した担当親。openのissueを作業中へ |
| pr | PRを作成・照合した担当親。`pr`に関連open PR番号を指定しレビュー中へ |
| stop / resume | 停止・再開の詳細をissueへ記録した担当親。現在の段階を維持し、理由を設定／解除 |
| merge | merge gate後に担当親。`pr`に実際にmergedの関連PRを指定。子はclosed/completedも必須。親は受入待ちへ |
| accept | 親が全体受入条件を確認した時。親だけに受入済み・完了を明示指定。未完了の直属子・取消以外の未完了末端があれば拒否 |

親（入口のissue、子を持つissue、または未分割の `issue:parent`）の行には `acceptance` を必ず指定する。
子の行には指定しない。`pr`はpr/mergeイベントにだけ付ける。
親の受入待ちの例は `stage: "受入待ち", acceptance: "受入待ち"`、受入は両方を完了・受入済みにする。
取消issueは `stage: null`、空でない取消理由・次の対応を指定し、成功の完了列に入れない。
stop/resumeは既存の段階と同じstageを指定する。停止前の段階を別の待機列で隠さない。

## 集計と完了の区別

- 正式な直属子を各親で1回だけ数える。closed/completedだけを実績の完了とし、not_planned・duplicateは分子・母数から除外する。
- 末端集計では子を持つissue自身を数えず、その子孫だけを数える。取消された枝は子孫ごと除外する。直属と末端を足し合わせない。
- 標準Sub-issues progressはAPIの値をそのまま併記する。標準側の母数・取消・階層の扱いを上書きしたり、除外後の実績と同じと仮定したりしない。
- 親本文のチェック数・母数・正式関係との差・チェック漏れをinspectに出す。本文だけの番号を正式な子に追加せず、チェック漏れを未実施に置換しない。旧書式の解析失敗も明示する。
- PRのopen・merged・closed_unmergedを分ける。PR関連判定は既存parentと同じCloses/Fixes/Resolves指示を使う。単なる言及やbranch名を証拠にしない。標準Linked pull requestsは別途そのまま表示する。
- 子のPR merge、GitHub issueの完了、親の全体受入は別状態。PRが未マージでclosedでもmergeイベントは拒否する。PRがmergedでも子がまだopenなら完了にしない。
- 全子がclosedになっても親の受入は変えない。研究テーマなど、親issueのcloseは従来の人間の判断に残す。

追加の進捗項目・台帳は作らず、親の「次の作業」の末尾に直属・末端・標準・本文チェック数を付ける。
子だけを更新した節目でも親の集計を再取得する。指定された次の作業と親の受入状態は勝手に変えない。
末尾の `進捗（自動集計）:` は予約済みで、入力のnext_actionに含めない。

## 標準APIの制約とUI作業

現APIの [create/updateProjectV2View](https://docs.github.com/en/graphql/reference/projects#updateprojectv2viewinput) は
ビュー作成・layout・filter・visibleFieldIdsを扱える。configuration入力にグループや列の指定はない。
カンバンの初回作成には [RESTのビュー作成](https://docs.github.com/en/rest/projects/views#create-a-view-for-a-user-owned-project) の
`group_by` / `vertical_group_by` を使い、Parent issueグループと段階列も設定する。
RESTへ渡すユーザー・項目の数値IDはGraphQLで取得する。API versionは2026-03-10。
ユーザーProjectのこのREST入口はclassic tokenのProject権限を使い、fine-grained token／GitHub Appは非対応。
失敗時に別の認証やGraphQLによる列未設定の作成へ切り替えない。

既存ビューのグループ更新口は同REST文書では提供されていない。
既存カンバンの列・グループが期待値と違う場合はmanual_stepsで次のUI修復を案内し、ビューを削除・再作成しない。

1. 「子タスクのカンバン」を開く。View → Column field → 段階を選び、6列すべてを表示する。
2. View → Group by → Parent issueを選ぶ。変更を保存する。
3. 必要なら親フィルタで #1204 に絞り、親一覧・待ち一覧も含めて表示を照合する。
   初回導入も表示を確認する。setup再実行はグループ・列を読み取り、期待値と違う場合だけmanual_stepsへ残す。

列とグループの操作は [GitHub公式手順](https://docs.github.com/en/issues/planning-and-tracking-with-projects/customizing-views-in-your-project/customizing-the-board-layout) を参照する。
Viewの標準Sub-issue progress自体に取消除外後の独自母数を設定するAPIは使わない。差は上記の集計欄で説明する。

GraphQLのworkflowは名前・enabled等のメタデータだけで、ルールの対象項目・値・auto-addフィルタの取得／変更口はない。
親／管理者がProjectのメニュー → Workflowsで既存ルールを確認する。
Item closedやPR mergedを新しい段階・親受入へ一律に接続しない。
特にAuto-close issueを段階の完了へ接続するとissueのcloseまで誘発し得るため、既存Statusの対象をUIで確認する。
このツールはworkflowの変更・無効化を行わず、既存Statusも書き換えない。
[auto-add](https://docs.github.com/en/issues/planning-and-tracking-with-projects/automating-your-project/adding-items-automatically) は
既存issueを一括登録しないので、initialによる明示的な初期登録を省略しない。

## 読み取り専用PMと失敗時

```bash
PYTHONPATH=.agents/ops .venv/bin/python -m project_progress inspect --repo Motoki0705/tennis-lab --parent 1204
```

inspectは認証中のghでProject設定・登録状況・正式関係・PR・集計を読み、JSONで返す。
親のidentity・registry・lockは不要で、stateを作らない。PMの定期確認はこの読取りまでとし、
Project更新・担当予約・agent起動・レビュー・merge・timerの変更へは接続しない。

成功0、取得／設定／整合エラー2。全ページを取得し、部分GraphQLエラー、null、重複、型・項目・選択肢不足、
権限不足を失敗にする。項目不足を既存Statusで補わない。cross-repositoryの子は明示調査を求める。
archived itemは勝手に再追加せず、UIで明示的にunarchiveしてから再実行する。

親registryの既存lockで同一ホストの書込みを直列化する。syncは最初の書込み直前にissue/PR/Projectを再取得し、
変化していたら拒否する。GitHub全体をatomicには更新できないため、途中失敗・応答喪失後は同じspec/eventで再実行する。
フィールド・ビュー名とcontent IDを照合して不足分だけを更新し、最終再取得で収束を確認する。
既存同名項目の型・選択肢が違う場合は上書きせず止め、UIで内容を照合する。

fake gh・一時registryによる試験は `tests/e2e/agents_ops/test_project_progress.py`。
この実装の試験は実GitHubへの書込みや本物のagentを必要としない。
