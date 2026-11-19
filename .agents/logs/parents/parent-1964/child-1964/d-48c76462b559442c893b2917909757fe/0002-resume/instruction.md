# PR #1971 統合レビューの major 2件の修正

統合レビュー run `parent-1964-530933e8afba410e8def8b5458d9777a`（対象HEAD 02ca25b85）で、次の2件の修正を求められた。
どちらも静的解析による指摘だが、親は妥当と判断した。修正し、それぞれの境界を再現するテストを追加すること。
テストは、修正を戻すと失敗するものにする。

## pr-1971-1 本文編集で取り出したparentの承認拘束を解除できる（`.agents/ops/parent/improvements.py` 407行付近）

**指摘:**
- 取り出したparentでは、初回の agree/basis-record より前に本文の出典行を削除できる。このとき履歴に improvement_approval がないため、require は None を返す。
- dispatcherの収集・guardも、本文の目印だけで出典を判定している。そのため、生成parentが通常のparentとして扱われる。
- 例: 取り出し後、担当登録前に候補を revoke してから出典行を削除する。すると停止要求を保存するregistryもなく、撤回済みの承認を照合しないまま自律実行根拠を作れる。

**直し方の方針（親の判断）:**
- 取り出し記録（共有private stateの取り出しreceipt、または証拠用Issueの正本記録）を「取り出し先parent番号 → 候補・承認decision_id群」の索引として持つ。
- 親CLIの全実行入口と dispatcher/supervisor の照合では、本文の目印ではなく、この番号の索引で拘束の有無を判定する。対象の入口: agree・basis-record・create-child・delegate・run-worker・start/continue-worker・authorization・dispatcher収集[local path]。
- 本文の目印と索引が食い違う場合は、明示エラーで止める。食い違いの例: 目印の消失・改変・別候補IDの記載。
- 本文の手編集で拘束を外せないこと、初回合意前の目印消失を拒否することを、テストで固定する。
- 索引の取得に失敗したときは「拘束なし」に読み替えず、失敗として止める。

## pr-1971-2 提案掲載の途中失敗からの復旧不能（`.agents/ops/parent/improvement_proposals.py` 343行付近）

**指摘:**
1. 候補A・Bの初回proposeで、Aのproposed記録だけが成功して中断する。
2. その後Aを更新すると、improvements.record は proposal 記録済みのAの更新を許可する。
3. 再送時はBが未記録なので pending になる。このループはAにも receipt の旧revisionとの一致を要求するため、毎回拒否される。
4. revisionは単調増加するので、内容を戻しても復旧できない。

**直し方:**
- レビューが挙げた2案のどちらかを選ぶ。
  - 案1: 未記録の候補だけに revision 固定を適用する。
  - 案2: 掲載処理全体の復旧が完了するまで、対象候補の更新を拒否する。
- 選んだ理由を最終報告に書く。
- 親の希望は、承認の固定対象（候補revision）と提案Issue本文の整合が崩れない方。部分成功後の更新、再送による収束、重複Issue 0件をテストで固定する。

## 共通

- 修正したらpushし、対象HEADのCI `Python` を確認する。
- 最終報告には次を含める:
  - 修正commit・HEAD
  - 各指摘への対応と、追加テストの名前
  - 修正を戻すと失敗することの確認方法
  - 関連テストの結果
  - CI結果
