探索CLIのPR #2044 がmainへmergeされました（merge commit 89040e5a6）。PR #2043 の統合レビュー（run parent-2011-ed68cc4ba6e14bd68c1ce0aefab2cce8）の指摘は2件です。PR内の指摘（created_at・URLの境界条件）は 734536103 で対応済みです。もう1件の横断指摘が残っているので、同じbranch・PRで次を進めてください。

## 横断指摘（major）

`knowledge/webui/src/lib/citation-documents.ts` が、#2044で拡張された保存契約を受理しません。

- 中断した探索の `status` / `stop_reason` を未知キーとして拒否する。
- 中断した探索の部分展開（最後の世代の expanded_ids が paper_ids の部分集合）を不整合として拒否する。
- readCitations が全体を ok:false にするので、中断した探索が1件あるだけで、完了した探索も閲覧できなくなる。

## 作業

1. branchを最新のmainへrebaseしてください。kg_citations.py・kg_identifiers.py・kg_schema.py などは両PRで変更しているので、#2044の変更と 734536103 の境界条件の対応を、どちらも失わずに統合してください。
2. 保存契約の正本は main の `knowledge/README.md` です。TSの型・検証・状態表示を、そこに書かれた規則にすべて合わせてください。少なくとも次の点です。
   - 探索の `status`（complete / interrupted）と `stop_reason`、中断した探索の部分展開の規則
   - 参照行の `identification_status`（resolved / unresolved / pending / unverified）と任意項目の `lookup`
   - pending / unverified が残る論文は展開されないという規則
   - 他にも #2044 で追加・変更された項目があれば同じく揃える
   - PythonとTSで受理・拒否の条件を変えないでください。
3. 閲覧画面では、中断した探索とその中断理由、照会待ち・一意性未確認・未同定・判定不能・本文取得失敗を、互いに区別して表示してください。1件の文書が不正でも、エラーをどう表示するかは既存の方針に従ってください。ただし、正しい完了済みの探索が、中断した探索があるだけで見えなくなることがあってはいけません。
4. テストは、main にある共有fixture（`tests/fixtures/knowledge/citation_map/` の中断・照会待ちを含む例と、`citation-boundaries.json`）をPythonとTSの両方で使い、受理・拒否の結果が一致することを確認してください。加えて、main の実データ `knowledge/citation_map/`（WASBの115論文・2探索）を、TS側がエラーなしで読めることをテストで確認してください。
5. 実データのWASB「比較対象・深さ2」を開いたマップと、比較対象のランキングのスクリーンショットを `knowledge/webui/assets/` に追加し、PR本文に載せてください。親が最終受入で照合します。
6. `knowledge/webui` で lint・test・build・e2e を、リポジトリ側で関連するpytestを実行してください。push後にCIの成功を確認して報告してください。
7. PR本文の先頭行には `Closes #2024` を入れてください。現在の本文にはありません。本文を書き換えるときも、この行は必ず残してください。
