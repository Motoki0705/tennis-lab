## 親からの継続指示（正式レビュー前の修正：中断runが同じPRを永久に塞ぐ）

PR #1872 の差分を親が確認した。正式レビューの前に、次の1点を同じbranch・PRで修正する。

### 問題
`review_locks.require_finished_overlaps` は、同じPRを含む別runの保存executionが `running` なら新規run・recover・resume-publicationを拒否する。`running` が残るのは、parent_review本体のプロセスがクラッシュした場合（WSL停止・SIGKILL・OOMなど）だけである。このラップトップでは実際にWSLが停止している。

ところが `running` から抜ける経路は、`resume-publication` だけである。これはoutputが `validated` で、postcheckが `passed`、かつ `result.json` があることを要求する（`parent_review.py` 920付近）。そのため次の中断では、そのPRの新しいレビューを二度と起動できなくなる。state JSONを手で編集するしかなくなる。

- 準備中またはagent実行中の中断
- final保存前の中断
- postcheck前の中断

本PRの前は、別runで再レビューできた。これは排他の導入による後退である。PR本文の「復旧条件を満たさない中断は保留します」も、解除手段がなければ恒久的な停止を意味する。

### 求める修正（設計は安全側で決め、READMEに契約を書く）
- 中断runを明示的に終結させる、親のコマンドを追加する。例: `parent_review.py close-interrupted --run <id> --reason-file F`。終結後は、そのrunが同じPRの新規起動を塞がないようにする。
- 時間やPIDからの推測で解放しない方針は維持する。終了は次の証拠で確認する。
  - そのrunの `run.lock` と、対象PRのleaseを非待機で取得できること。これで本体プロセスの終了を示す。
  - agent工程が開始済み（`agent_invocations=1`、またはagent stageが `not_started` 以外）の場合は、共通ランナーの `agent-runtime/result.json` で `cleanup_complete=true` を確認する。この記録は、ランナー自身が書いたものでも、`agent_exec.sh stop --cleanup-confirmation` で親が確認したものでもよい。runtime記録が無い・未完了なら拒否し、必要な手順をエラーで示す。
  - review workspaceが残っていれば拒否する（または既存の撤去手順を示す）。
- 終結の記録は正式結果として承認・coverageに使われてはならない。failed/not_reviewed相当の結果とする。既存の公開記録と整合させるため、既存の `publish_result` で公開するか、公開しない終結状態にするかを選ぶ。どちらでも、gate・`superseding_reviews`・`review-inspect` での扱いを一意に定める。保存済みfinalがあり、復旧条件を満たす場合の既存の `recover` / `resume-publication` の経路は壊さない。
- `review-inspect` にも、中断・終結の状態が機械的な事実として表示されるようにする。
- テスト（`-n 0`、対象nodeのみ）:
  1. agent実行中に中断（`running`、runtime未完了）したrunは、終結を拒否する。
  2. runtimeのcleanup確認後は終結でき、その後に同じPRの新runを起動できる。
  3. 終結したrunの結果が、gateの承認・coverage・後続requestによる置換の判定で正式な承認として扱われない。
  4. 本体プロセスがまだ `run.lock` を保持している間は、終結できない。
- 故障注入として、cleanup確認の検査を外す変異を1件入れ、検出されることを確認する。

### CIについて
親がrun 42832676759 の失敗ジョブ（レーン1・7、apt timeout、#1857）を1回再実行した。修正をpushすると新しいCIが走る。aptの準備失敗は、盲目的に繰り返し再実行しない。PR本文のCI欄は、新しいHEADの結果で更新する。

### 共通
- 同じ種類の抜け道（終結記録を承認に使える経路、終結で他runの未解決指摘を隠せる経路）が無いかを見直す。
- PR本文の契約・検証表を更新する。最終報告では、設計の選択（公開するかどうか）、追加テスト、故障注入の結果、新HEADのCI状態を返す。作業時間の目安は最大2時間。
