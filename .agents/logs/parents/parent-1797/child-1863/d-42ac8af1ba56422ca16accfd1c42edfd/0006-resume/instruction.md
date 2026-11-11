## 親からの継続指示（PR #1907 の正式レビューの指摘修正）

PR #1907（head 3de0c95）の正式な親レビュー（run `parent-1797-438ceae2651340cb86fb3e26a80e6c00`、fixed Astra・xhigh）は request_changes で、major 4件だった。親は4件とも妥当と判断した。同じbranch・PRで修正する。いずれもレビュー側の静的推論なので、まず再現テストを書き、失敗を確認してから直す。

### pr-1907-1（major）非公開取得サーバーの全入口で、正式stateへの捕捉を遮断する
`.agents/ops/review/evaluation/__main__.py` の73行付近。`serve` を直接起動すると、継承した有効な `DECISION_LOG_RETRIEVAL` により、`FrozenRetrieval` が継承する `Retrieval.call` から `decision_logs.observations.retrieval` が呼ばれる。その結果、正式な parent journal へ比較用の取得結果が書き込まれる。環境変数の除去は execute 経路にしか無い。
- 入口に関係なく、サーバー側（`FrozenRetrieval` またはその起動部）で、正式な捕捉を確実に無効化するか、捕捉の設定が残っていれば起動を拒否する。どちらにするかは、静かなフォールバックにならない方を選び、READMEに書く。
- テスト: 有効な `DECISION_LOG_RETRIEVAL` を設定した状態で `serve` を直接起動しても、正式journalへ何も書かれないこと（または明示的に拒否されること）。

### pr-1907-2（major）親判定の分類漏れと重複を拒否する
`.agents/ops/review/evaluation/metrics.py` の205行付近。`unknown += len(set(emitted) - classified)` により、other_findings から分類が漏れた指摘を自動で unconfirmed へ加算し、parent_adjudicated・scored として受理している。また、matches 内の同じ対応の重複が set に吸収されて受理される。
- 次の判定は、受理せずに拒否する。
  - match にも other_findings にも無い指摘がある（分類漏れ）。
  - 同じ finding_id が matches と other_findings の両方にある。
  - matches 内に同じ (finding_id, gold_id) の重複がある。
  - 1つの finding が複数の gold に対応している（契約上許すなら、READMEに根拠を書いたうえで別扱いにする）。
- 親の未記入と、明示的な unconfirmed を区別する。
- テスト: 上の各場合の拒否と、現行の adjudications.json が引き続き受理されること。

### pr-1907-3（major）runtime失敗を、比較の検出成功へ計上しない
`.agents/ops/review/evaluation/execution.py` の308行付近と `metrics.judged`。有効な完了finalがある cli_error・runtime_error の failed 実行も、runtime_evidence が復旧可能として返す。その場合、output_status は valid、runtime_status は failed になる。ところが metrics.judged は runtime_status を見ないため、親が match を記入すると失敗runも true_positives に入る。
- 保存候補の保持（後から調べられること）と、比較の成功検出（集計に入ること）を分ける。runtime_status が成功でないrunは、検出数・誤指摘数の母数に入れず、runtime失敗として別に数える。
- テスト: 有効finalを持つ runtime失敗のrunに match を付けても、true_positives に入らず、失敗として数えられること。

### pr-1907-4（major）公開資料へ実worktreeのパスを残さない
`.agents/ops/review/evaluation/cases/inputs/pr-1686.json` の7行目と97行目付近。当時の issue 本文から写した `.claude/worktrees/...` が、公開した固定入力に残っている。
- 公開する固定入力から、worktree のパス（および絶対パス・session ID）を除く。比較の来歴を壊さないように、次を守る。
  - モデルが実際に見た元の入力の digest は、run記録・provenance に保持する。
  - 公開版が元の入力を無害化したものであることを、置換の規則・元のdigest・公開版のdigestとともに記録する。
  - `review.evaluation verify` が、無害化後も来歴を検証できる。
  - 実LLMの再実行はしない。
- 公開検査（測定結果に対して行っている検査）を、固定入力・判定資料・READMEなど、このPRで公開するファイル全体へ広げる。他のケースにも同じ問題が無いか確かめる。
- テスト: 公開検査が、固定入力に含まれる worktree のパスを検出すること。

### 共通
- 同じ種類の抜け道が他に無いか、変更範囲を見直す。
  - 評価経路から正式stateへ書き込む経路。
  - 判定・集計で、欠落や失敗を成功扱いにする経路。
- 故障注入を、各指摘1件以上行う。
- 修正後に report の parent-adjudications を再実行し、公開した数値・結論が変わらないか確認する。変わった場合は、変更点と理由を結果READMEとPR本文に書く。
- pytest は `-n 0` で、対象のnode・ファイルだけを順に実行する（RAM約3.9GB）。
- CIの基盤由来の失敗は、ログを確認したうえで、失敗ジョブの再実行を1回までにする。
- 実LLM・実agentは起動しない。GitHubへの書込みは、このPRのpushと本文の更新だけとする。
- PR本文の契約・検証表を更新する。
- 最終報告では、次を返す。全角スラッシュ「[local path]」は使わない。
  - 各指摘への対応、追加テスト、故障注入の結果
  - 再計算の結果
  - 新HEADのSHAとCIの状態
- 作業時間の目安は最大3時間。
