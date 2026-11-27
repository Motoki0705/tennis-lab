# レビューとマージ

## 依頼する前に

人間がレビューするPRは、検証結果と残る判断を付けて人間へ渡す。自動マージを任されたPRでは、`AGENT_AUTO_MERGE=1`、担当・実行根拠、対象Issue・PRの `needs-human` と停止状態を確認する。自動進行自身がこれらの制御を変更することはない。

親は [wait-worker](workers.md#待つ追加する受け入れる) で統合担当の終了と回収を確認し、PRのclosing対象・branch・HEADを元の受入条件と照合する。ローカル検証済みのready PRは、PR CIの完了前に `review-request` を出し、正式レビューとPR CIを並行して進める。ドラフトは準備が整ってからreadyにして依頼する。

原要件・影響先を `--instruction` または `--instruction-file` で渡す。対象HEADの検証結果・未確認事項は `--evidence-file -` で標準入力から、保存済みnoteは `--evidence-ref` にSHAまたはURLで指定する。依頼と同時に成果を記録する。`review-request --prs … --wait` で依頼すると、最初の通知または完了まで待つ。正式reviewerはSol。複数PRをまとめる場合も、各PRの正式レビューは有効な結果1回で確定し、依頼の再送には既存runが返る。

## 記録専用PRをマージする

[作業者が分けた記録PR](workers.md#ナレッジ記録prの作成) は `review-inspect --prs …` の `review_candidates` から除外される。`review-request` でも専用操作を案内する。比較を確定できないPRは正式レビュー候補に残り、理由は `scope.reason` に出る。

親は対象Issueの受入条件をPRの現差分と照合し、[PR本文の実行証拠](workers.md#ナレッジ記録prの作成) を確認する。実行が必要ならコマンドと結果を現HEADで確認し、`execution` を `verified` とする。実行を伴わない受入は `not_required` とする。選択した各PRの番号・完全SHA・確認結果を次の形式で保存し、`merge-knowledge --prs 1250 --evidence-file verified.json --wait` で渡す。保存済みnoteには同じJSONを記し、`--evidence-ref` で参照できる。

```json
{
  "schema_version": 1,
  "prs": [
    {
      "pr_number": 1250,
      "head_sha": "0123456789abcdef0123456789abcdef01234567",
      "acceptance": "verified",
      "execution": "verified",
      "summary": "指定した評価32件の出力をPR本文のログと照合した。"
    }
  ]
}
```

CLIは本文の閉鎖指定とGitHubが認識する閉鎖対象を照合し、すべてが担当範囲内であることを確認する。PRタイトルやコミットメッセージから閉じるIssueも本文に明記する。別リポジトリへの閉鎖、未解決の指定、本文にない閉鎖対象、実装付随PRの閉鎖指定があれば保留する。現HEADと最新mainで成功した [Knowledge library](../../../ops/lib/README.md#ナレッジ記録専用prのci) を待つ。マージ直前に、実行根拠、`AGENT_AUTO_MERGE`、`needs-human`、停止指定、閉鎖対象とPR本文を再照合し、同じHEADと比較元の成功run・attemptを確認する。

待機と続行には同じrun IDで `merge-status`、`wait-merge`、`merge-continue` を使う。mainが進んだら、新しい比較元でのPRイベントの検査結果を待つ。続行するPRのうちマージ送信前のものについて現HEADと最新の本文を確認し、同じ形式のJSONを `merge-continue --run-id RUN --instruction-file verified.json --wait` に渡す。マージ応答喪失やmain CI待機の再照合だけを続けるときは、同じrunに続行理由を渡す。

保留中のPRが実装混在に変わったら、`review-inspect` で候補になったことを確認し、通常どおり `review-request` を行う。CLIはPRがopenで未マージ、専用controllerが停止済み、マージ送信の記録がないことを確認したうえで、そのPRの専用予約を `transferred` として終え、正式レビューに渡す。送信済みの場合は専用runでマージ結果を照合する。

マージ後はmerge commit、mainの `Python`、関連Issueの状態を `status.completion.knowledge_merges` で参照できる。Issueを閉じるPRでは `closed/completed` の確認が必要。実装PRに付随する記録PRでは、参照したIssueの観測状態を表示する。

## 通知を受けたら

手動親には、登録済みのClaude socketまたはCodex queueへ自動進行の呼び鈴が届く。Claudeとagyではhookにも未確認件数が出る。通知のrun IDを使い `merge-status --run-id RUN` でイベントと配送状態を確認する。 [手動親への通知](../../../ops/review/README.md#手動親への通知)

| 観測 | 次の操作 |
|---|---|
| approve、`ci_wait`、`dependency_wait` | CLIが現HEADのPython CI・先行Issue/PRを確認する。待ちの間は独立作業を進め、次の通知を `wait-merge --run-id RUN --after 通知ID` で受ける。 |
| major | 採用した指摘を統合担当へ渡す。親は `wait-worker` の終了後、修正commitとローカルの回帰テストまたは再現結果を元の指摘と照合する。対象外の指摘には理由を記す。受入条件の変更が必要なら人間に確認する。全件を処理したら、修正後のPR CI完了前でも同じrunに `merge-continue --run-id RUN --instruction-file fixes.md --wait` を出す。controllerが新HEADのPR CIを待ち、失敗時は作成ワーカーの自動修正に進む。 |
| CI失敗・競合 | CLIは登録された作成ワーカーの同じsessionを修正ターンのhighで再開し、push後は新HEADのCIへ戻る。自動修正はPR単位でCI・gate合算3回。作成ワーカーがない場合や上限に達した場合は、親が失敗箇所と残作業を判断する。 |
| `parent_wait` | `merge-status --run-id RUN` のイベントと理由を読む。停止・権限不足は解除条件を確認し、起動やマージの応答喪失は [保存状態から復旧](recovery.md) する。 |

一時障害で正式レビューの結果が出なかった場合、`merge-continue --retry-review ROUND` を同じrunに明示して再試行できる。[終了・回収の照合と再試行上限](../../../ops/review/README.md#正式レビューの障害復旧)を参照する。

`merge-continue --prs` で一部のPRを再開するときも、controllerは同じレビューrunに含まれる全PRのリースを取る。手動ワーカーの監視と競合したら、表示されたPRの `wait-worker` の結果を照合してから続行する。起動後の競合は `parent_wait` の通知に対象PRと理由が残る。

approveまたはmajor処理後、現HEADのCI・依存・マージ条件が揃えば、そのHEADを指定してマージする。マージ後は**merge commit**のmain CIと対象Issueの `closed/completed` を確認する。この後確認と、親全体の [受入](finish.md) は別の記録になる。

回数・時間・コストを調べるときは `merge-status.review_usage` を読む。各回の起動、延べ実行時間、入力・キャッシュ済み入力・出力と、PRごとのマージ観測までの経過時間を区別する。
