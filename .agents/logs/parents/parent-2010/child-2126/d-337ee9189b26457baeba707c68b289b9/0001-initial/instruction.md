# #2126 の実装（PR作成まで）

担当Issue #2126 の本文が要件と受入条件の正本です。PR #2116（#2114）で merge gate の読み取りを観測ごとに絞りましたが、正式レビューの major pr-2116-1 を未修正のまま、人間の判断で main に merge しました。その修正です。

## 指摘（pr-2116-1）

`.agents/ops/review/merge_gate.py` の merge 直前の採用照合（`check_adoptions(session, fresh.details[n], (fresh.parent, fresh.children))`）に渡す fresh.parent / fresh.children は、レビュー証拠取得より前に収集した本文です。その後のPRコメント取得中などに子Issueの採用binding（参照先・option・proposal_sha256）が不整合な値へ変更されても、fresh_reads() が再取得するのは古いbindingの参照先reportだけなので、変更を検出せずmergeできます。変更前はここで work_items() が最新本文を取得し、不整合を拒否していました。正式family全体の再探索は省いても、親とclosing対象Issueの本文は最終照合で再取得してください。追加テストはreport側の取消だけを注入しており、本文側のbinding変更を検出しません。

## 依頼

1. merge 直前の採用照合で、親 Issue と、その PR が閉じる子 Issue の本文を fresh に再取得し、最新の binding で照合する。正式 family 全体の再探索は不要。照合に使う内容が変わっていたら merge しない。
2. 同じ「収集時の値を merge 直前の照合に使っている」型の箇所がほかにないかも確かめ、あれば同じく直す。
3. 回帰テストを足す。レビュー証拠の取得の後・merge の前に、子 Issue（と親 Issue）の採用 binding を不整合な値へ変えたら merge しないことを確かめる。修正を外すとこのテストが失敗することを確認する。要求数の上限テストは、増えた分だけ見直し、理由をテストに書く。
4. 関連テストと pre-commit を通し、`Closes #2126` 付きで PR を作り、対象HEADの CI（Python）の成功を確かめる。

人間の判断で、正式レビューは PR ごとに1回だけです。このPRは小さな修正なので、親があなたの報告・差分・回帰テスト・CI を確かめて merge します。最終報告には、どう直したか、どのテストがそれを検出するか（修正を外すと失敗することの確認を含む）を具体的に書いてください。

共有の GitHub API 上限は逼迫しがちです。検証は fake GitHub で行い、実 GitHub は push・PR 作成・CI の確認だけに使ってください。短い間隔の polling はしないでください。時間の上限は3時間です。
