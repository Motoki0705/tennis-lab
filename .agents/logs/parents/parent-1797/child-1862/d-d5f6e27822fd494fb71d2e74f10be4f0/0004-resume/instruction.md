## 親からの継続指示（正式レビューの指摘修正）

PR #1888 の正式な親レビュー（run `parent-1797-b066652105a247609734d38fd3eab66f`）は request_changes で、major 2件だった。親は両方を妥当と判断した。同じbranch・PRで修正する。どちらもレビュー側の静的推論なので、まず再現テストを書いて失敗を確認してから直す。

### pr-1888-1（major）採用可能なreviewed結果には、run全体のstatusに関係なく実効モデルの検証を必須にする
`.agents/ops/review/review_recovery.py` 347付近。複数PRのrunで、一方がreviewed、他方がnot_reviewedの場合を考える。run全体はfailed・incompleteになるが、`evaluate_coverage` はreviewed側のmergeを許す。ところが、この分岐はfailed結果のruntime照合を省略する。そのため、保存runtimeの実効model・effortの不一致やdigestの変更を、gateと公開再開で検出できない。
- 「run全体がsucceededか」ではなく、「gate・公開再開・recoverで採用されうるreviewed結果があるか」を基準にして、同じ証拠検証（選択値と実効値の一致、runtime digest）を必須にする。
- テスト: 2PRのrunで一方がreviewed、他方がnot_reviewedのとき、実効modelの不一致やruntime digestの改変を、gateと公開再開が拒否すること。既存の改変検出テストはsucceeded結果だけを扱っているので、この境界を追加する。

### pr-1888-2（major）handoffで、保存済み候補の指摘を脱落させない
`.agents/ops/parent/review_choice.py` 59付近。有効なfinalを保存したあとに、通信障害や事後照合の失敗が起きることがある。この場合、major指摘は `output.json`（候補）に残るが、`result.json` は指摘なしのfailed結果になる。`source()` は `result.json` だけを読み、次runの `review_handoff` にもそれだけを渡すため、保存された指摘が引継ぎ資料から消える。後続runがapproveすると、gateは最新runを採用し、元の指摘を照合しない。
- handoffの元runに、検証済みの保存候補（output.json、子2の `review-inspect` が `candidate_only` として扱うもの）があれば、その指摘・証拠待ちを、正式coverageとは区別した候補として引継ぎ資料に固定する。固定先はrequest digestに含め、レビュー用プロンプト・contextにも渡す。
- 候補の指摘を正式な未解決指摘としてgateで扱うかどうかは、安全側で設計してREADMEに書く。少なくとも、後続runのレビュー資料から黙って消えないこと、また候補が正式coverage・承認の根拠に使われないことを保証する。
- テスト: 「指摘付きfinal＋公開または事後照合の障害」の元runからhandoffしたrequestに、候補の指摘が固定されること。候補が無いrun、候補が不正なrunの扱いも確認する。

### 共通
- 同じ種類の抜け道（failed扱いの結果を経由して検証を省略する経路、引継ぎで指摘が消える経路）が他に無いか、変更範囲を見直す。
- 故障注入を各指摘1件以上行う。
- pytestは `-n 0` で、対象のnode・ファイルだけを順に実行する。メモリ不足の再発を避ける。
- CIの基盤由来の失敗は、ログを確認したうえで、失敗ジョブの再実行を1回までにする。
- PR本文の契約・検証表を更新する。最終報告では、各指摘への対応、追加テスト、故障注入の結果、新HEADのCI状態を返す。全角スラッシュ「[local path]」は使わない。作業時間の目安は最大3時間。
