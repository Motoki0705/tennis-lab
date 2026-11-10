## 親からの継続指示（正式レビューの指摘修正）

PR #1872 の正式な親レビュー（run `parent-1797-54a389ff62c7475391c32bd01ee00a3b`）は request_changes で、major 2件だった。親は両方を妥当と判断した。同じbranch・PRで修正する。どちらもレビュー側の静的推論なので、まず再現テストを書いて失敗を確認してから直す。

### pr-1872-1（major）終了・cleanupが未確認のrunを、公開状態だけで「終わった」とみなさない
`.agents/ops/review/review_locks.py` 78付近。`require_finished_overlaps` は `running` だけを拒否している。しかし `_run_review` は、`runtime_evidence` の失敗（agent stageが `unconfirmed`）や `workspace_cleanup_complete=false` を捕捉したあとも `publish_result` へ進み、executionを `published` / `publication_failed` にする。共通ランナーだけがSIGKILLなどで終了し、CLIが残った場合、lease解放後に同じPRの別runが起動できてしまう。
- 排他の判定では、executionのstatusに関係なく、agent stageの終了確認（`finished`、`cleanup_complete=true`）とworkspace撤去が揃っていないrunを阻害条件にする。旧形式の記録で、stageが無いものの扱いは安全側で一意に定める。
- この状態から抜ける明示経路を用意する。例: `close-interrupted` を拡張し、cleanupの証拠を確認したうえで、公開済みのrunにも「終了確認済み」の記録を追記する。公開済みの正式結果（指摘・coverage）は書き換えず、gate・resolve・`superseding_reviews` の判定を変えない。設計は安全側で決め、READMEに契約を書く。
- テスト: published / publication_failed のまま、agent未確認またはworkspace残存のrunがあると同じPRの新runを拒否すること。終了確認後は起動できること。既存の「published / publication_failed を無条件に許容する」テストを修正する。

### pr-1872-2（major）runtime記録が作られる前の中断にも、検証付きの終結経路を用意する
`.agents/ops/review/review_interruption.py` 98付近。`parent_review.py` は `agent_invocations=1` と `agent.status=running` を保存してからrunnerを起動する。この後、`agent-runtime/result.json` が最初に保存される前に本体が止まると、終結できない。案内している `agent_exec.sh stop` も、`--cleanup-confirmation` を処理する前に同じ `result.json` を必須で読むため、この経路は使えない。
- まず、共通ランナー（`.agents/ops/lib/agent_exec.sh` と、その状態記録）が起動のどの時点で何を書くかを確かめ、この窓が実在するかを再現テストで示す。
- 実在するなら、記録の欠落を安全とは推定しない。親が確認した証拠を受け取って終結する経路を作る。例: `close-interrupted --cleanup-confirmation <JSON>` で、共通ランナーの既存のcleanup確認形式（schema_version・run_id・runner・process・cleanup_complete・evidence）に揃える。その証拠をexecutionの終結記録へ保存する。共通ランナー側を直す方が筋がよければ、その変更でもよい。ただし `.agents/ops/lib` は下流が広いので、既存テストを壊さないことを確認する。
- テスト: runtime記録が無い中断で、確認証拠なしの終結は拒否されること。証拠があれば終結でき、その後に新runを起動できること。

### 共通
- 故障注入を各指摘1件以上行う（それぞれの新しい検査を外し、テストが検出することを確認する）。
- pytestは `-n 0` で、対象のnode・ファイルだけを順に実行する（WSL停止の再発防止）。全体はCIで確認する。
- CIのapt取得失敗（#1857）は、ログを確認したうえで、失敗ジョブの再実行を1回までとする。
- PR本文の契約・検証表を更新する。最終報告では、各指摘への対応と設計の選択、追加テスト、故障注入の結果、新HEADのCI状態を返す。作業時間の目安は最大2時間半。
