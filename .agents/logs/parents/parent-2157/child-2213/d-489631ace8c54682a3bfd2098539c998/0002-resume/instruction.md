# PR #2214 のレビュー指摘（major）の修正（#2213）

自動進行の大局レビュー（run parent-2157-4d2522e8a9ce4e7bbe63e65ab8510f2f）で major が1件出ました。修正して同じPR（branch `codex/completion-candidate-groups`）へpushしてください。

## 指摘: 生成側が許容する複数行の受入条件を完了判定でも扱う
completion.py:53–55 は、候補グループ内の続き行を無条件に拒否します。しかし acceptance=['保存できる。\n  再読込して内容が一致する。'] は既存の validate_spec を通り、extraction_draft（内部で _extraction_body を使用）も改行を保持した本文を正常生成しました。その本文と同一の合意snapshotで status を実行すると、requirements.available=false、理由は acceptance snapshot is not an explicit checklist となります。正規に取り出せる親が引き続き構造化finishを記録できず、#2213 の第1受入条件を満たしません。追加E2Eは単一行の条件だけを使うため、この不整合を検出できません。既存本文・snapshotを書き換えず、生成可能な続き行を扱い、独立した自由文の拒否と区別する回帰テストが必要です。

## 親の方針
- 生成側（`candidate_sections` / `_extraction_body`）が実際に書く続き行の形を受け付け、直前の項目の条件文へつなげてください（通常形式の続き行と同じ扱い）。
- 一方、グループ見出しの直後でまだ項目がない位置の自由文や、生成側が書かない形の行は、従来どおり明示的に拒否してください。続き行と独立した自由文を区別する根拠（生成側の形）をテストで固定してください。
- 回帰テストは、`validate_spec` を通る複数行の受入条件から `improvement-extract`（または extraction_draft）で本文を作り、その本文と合意snapshotで `available=true` と finish の受理を確認するe2eにしてください。修正を外すと失敗することも確かめてください。

push後に、修正内容・テスト結果・新しいHEADを報告して終了してください。CIの完了は待たなくてよい。続行（merge-continue）は親が行います。
