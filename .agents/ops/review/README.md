# レビューからmerge後確認までの自動進行

担当親が `parent review-request` を1回呼ぶと、切り離したCLIプロセスが対象PR全体のCodexレビュー、CI待機、軽いgate、merge、merge後確認を進める。
PRのopen・push・時間経過からレビューを始めるtimerや常駐serviceはない。
入口とモデル選択は [parent README](../parent/README.md#レビュー要求と自動進行) を参照する。

## レビュー

`parent_review.py` は親・対象Issueの原要件とコメント、PRの本文とコメント、レビュー時のhead、全文diff、read-only checkoutを収集し、共通ランナーを1回だけ呼ぶ。
番号で辿った親関係と保存済みの証拠用Issueを資料へ含め、合意・判断の原文を保持する。証拠用Issueと依存PR・そのCI headもMCPの取得範囲に入れる。
証拠bindingがない関連Issueはその欠測を `evidence_bindings` に示す。担当親のbinding欠落・不確定な作成・既知bindingの不整合はエラーになる。
モデル選択の方針は [正式レビューのモデル選択](../parent/README.md#正式レビューのモデル選択) が正本。
選択値と観測した実効値を保存する。実行失敗、cleanup未確認、JSON形式不正は承認へ変換しない。

結果は `approve` または `major` の一覧。各majorは対象PR番号、title、detailを持つ。
majorでは親へ通知し、親が作成workerへ修正を依頼してから `merge-continue` する。
明示続行時に、手動/dispatcherの終了・cleanup、元session・モデル/契約、委任の継続来歴を確認する。
同じworkerの継続で委任IDが変わった場合は新しい修正登録へ結び直し、親の修正確認を成果へ記録する。自動修正の消費回数は戻さない。
親が `--rereview` を付けたときだけ追加レビューする。CI修正・競合解消・head変更では再レビューしない。
親の判断材料としてレビュー時head、現在head、変更ファイル数・追加削除行数を表示する。取得が不完全ならその範囲を明示する。
PRへの結果コメントは表示用のコピーで、merge権限の正本には使わない。公開失敗は結果に残し、保存済みレビューを再実行しない。

取得ツールと隔離CPUテストは [review_tools](../lib/review_tools/README.md)。自動進行から渡した対象Issue/PRとcommitだけを取得でき、repo全体のIssue検索ツールは公開しない。
[比較harness](evaluation/README.ja.md) は過去資料を保持し、履歴を本番のmerge許可へ変換しない。

## 状態と再開

正本はparent state内の `progressions/<run ID>/state.json`。controllerの排他とPRのleaseはregistryとは別に持つ。
保存は0600のatomic replace。registryのlockは担当確認・worker予約・通知の短い更新に限り、CI・依存・API上限の待機中には保持しない。
PRごとの段階・head・CI観測・自動修正回数・merge意図・確認結果、レビューと親の続行判断、通知イベントを保存する。

| 段階 | 次の動作 |
|---|---|
| `requested` / review `running` | 対象全体を1回レビュー。保存済み結果は再利用し、起動済みか不明なら親へ返す |
| `ci_wait` | 現headのPython CIを60秒間隔、最大6時間待つ。head変更は新headのCIへ戻る |
| `dependency_wait` | 対象の依存Issue/PRだけを60秒ごとに確認。他のPRは続行する |
| `repair_pending` | 修正意図を保存して作成workerを再開。終了・cleanup・JSON報告とpushされたheadを確認する |
| `merge_pending` | 意図保存後にheadを指定してmerge。再開時は結果を照合し、同じmergeを再送しない |
| `post_merge_wait` | PRのmerge、対象Issueのclosed/completed、merge commitのmain CIを確認する。既存成果の `merge-confirm` もこの段階から始まる |
| `parent_wait` | major、人間の制御、修正上限、異常、`recovery_required` を親へ通知。明示続行を待つ |
| `completed` | 全対象のmerge後確認が成功し、完了を通知する |

rate limitは元段階と `resume_at` を保存する。HTTP 403/429の分類、ETag、304再利用は [HTTP取得層](../issues/README.md#github通信認証repo解決) に従う。
通常の権限不足を待機へ読み替えない。ETag cacheはプロセス内だけで、再開後は200から始まる。

プロセス中断・WSL再起動後は、親が `merge-status` で記録を確認して `merge-continue` する。
同じ `review-request` の再送は既存runを返すだけで、停止したプロセスを勝手に再開しない。
merge応答を失ってもPRのmergedとmerge commitを確認できれば後確認へ進む。確定できなければ `recovery_required` とし、再送しない。
workerの起動応答を失った場合も、同じ修正IDの保存予約とruntimeを照合する。確認できなければ新workerを起こさない。
レビューruntimeの終了・回収が不明なまま別レビューを起動しない。必要な確認は [共通ランナーの復旧](../lib/README.md#状態確認停止中断からの復旧) に従う。

## 軽いgate

PRごとに次の順で検査し、最初の不合格で止まる。

1. 手元のレビュー承認またはmajor対応済みの指示、`AGENT_AUTO_MERGE=1`、担当・世代。
2. 現headの最新 `Python` check runの成功。欠落・進行中は待機し、失敗は修正workerへ渡す。
3. 競合、draft、branch/repo/closing対象、default branchへのPR、headの変化。headが変わればCIへ戻る。
4. 親・対象Issue・PRの `needs-human`、実行根拠、対象作業の停止・採用元。
5. 依存Issueのclosed/completedとdefault branchへのPRのmerge。依存Issueの番号指定timelineで後続のopen PRも確認する。PRのない調査子はclosed/completedで満たす。
6. `gh pr merge --match-head-commit`。

CI失敗は自動rerunせず、毎回作成workerの同じsession・モデルを修正ターンのhighで再開する。
失敗run/attempt、job、step、URL、ログまたはログ取得不能の理由を渡す。
gate停止もworkerが修正か親への判断依頼を選ぶ。ただし `needs-human` と `AGENT_AUTO_MERGE` 無効ではworkerを起こさず、依存未mergeはCLIが待つ。
短絡した場合もworker起動前の人間の制御・権限検査を省略しない。
作成workerの有無にかかわらず、全対象Issueの本文・registryの停止を確認する。レビュー起動前と保存済みの修正/merge/後確認段階の再開でも人間の制御を確認する。

自動修正は **PRごとにCI修正とgate修正を合算3回**。4回目が必要なPRは親へ返し、他のPRは続ける。
workerは修正・push後にそのターンを終え、CI待機はCLIへ戻す。報告schemaと接続APIは [修正ターン](../parent/README.md#自動進行の修正ターン) が正本。
作成workerが未登録なら親へ返し、代替sessionを新設しない。

人間の制御は従来どおり。`AGENT_AUTO_MERGE` は未設定または0で無効、1で有効、他の値はエラー。
workflowがこの値を設定したり `needs-human` を外したりすることはない。
専用の [workerログ](../worker_logs/README.md)・[判断ログ](../decision_logs/README.ja.md) の公開merge例外はこの経路に混ぜない。

## 通知と完了

手動親は `review-request --wait` または `wait-merge --after <通知ID>` を会話の背景実行へ置く。
待機は読み取り専用で、最後に見た通知以後のmajor・判断要求・完了を返す。
dispatcher親には保存済みイベントIDで既存のwake要求を渡す。親がまだ実行中でも通知を保持し、待機へ移った後に再開する。
修正workerだけの終了では親を起こさず、自動進行が報告を受け取る。機械による確認は判断ログでもmachineとして記録する。

merge後は移動するmain先端を使わず、merge commitのpush CIを確認する。失敗やIssue未完了は親へ通知する。
全体の受入判断・完了通知・担当終了は親の仕事で、[開発workflow](../parent/development_workflow.md#待機完了担当終了) へ戻る。
旧schema 1〜3の要求と公開記録は `review-inspect` / historyで読めるが、自動進行の承認として使わない。既存registry・過去成果は書き換えない。

### 切替前にmerge済みの成果

`parent merge-confirm --prs N... --wait` は、担当親に属する既にmerge済みのPRを番号指定で後確認する。
merge commitの存在、default branchへのmerge、対象Issueのclosed/completed、merge commitのpush CIを確認し、privateな後確認記録を親の完了判定へ渡す。
旧レビューの承認とは結び付けず、履歴・registryの旧review行も変更しない。レビュー・修正worker・mergeは起動せず、open PRと `--rereview` は受け付けない。
確認だけなので `AGENT_AUTO_MERGE=1` は不要。担当・実行根拠・needs-human・停止は確認する。待機・通知・明示再開は通常の `merge-status` / `wait-merge` / `merge-continue` を使う。

## 検証と取得量

`test_merge_progression.py` はfake GitHub・agent・clockで状態遷移、中断、60秒/6時間、上限を検証する。
`test_progression_cli.py` は実CLI・切離しプロセス・実gitとfake HTTP/agentを通し、通知・人間の制御・完了を検証する。
`tests/benchmarks/agents_ops/merge_gate_reads.py` は対象PR数と無関係Issue数を変えてGETを計測する。
自動進行では保存済み証拠Issueを番号指定で取得し、repo全体一覧へfallbackしない。bindingがなければ `parent evidence --create` で先に確定する。
