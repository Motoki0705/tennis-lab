PR #2026 の統合レビューで major 2件の指摘がありました。両方とも親として妥当と判断したので、同じbranch・PRで修正してください。

1. pr-2026-1（kg_citations.py の外部ID結合）: 一致したIDだけで結合し、同じnamespaceで値が矛盾していても黙認している。例: 固定例のBは doi=10.5555/fixture-b, arxiv=2401.00002 だが、Papers側を doi=10.5555/conflicting-b, arxiv=2401.00002 にしても papers_join がBへ結合し、validate_library もエラーにならない。一致候補が1件でも共通namespaceの値を照合し、矛盾があれば明示的に解決されるまでエラーにすること。Papers↔探索の結合と、探索文書どうしの同一視（別名の集約）の両方で同じ規則を適用し、READMEの契約にも書くこと。このケースのテストを追加すること。

2. pr-2026-2（kg_papers.py の採録対象）: `discovery_reason` が generations.paper_ids に含まれる文献だけを許可しているため、search-fixture-comparison-depth2 で基盤として記録・表示される Fixture C を採録できない。親 #2011 の合意では、役割指定は再帰展開だけを制御し、人間は表示された探索結果から必要な文献を採録できる。その探索で展開した文献（各世代の expanded_ids）が引用し、同定済みのすべての引用先を採録対象に含めること。発見経路には引用元・ref・役割・判断状態・根拠を記入し、複数の引用元から到達できる場合はすべての経路を記入すること。探索範囲外の文献（その探索で展開した文献から引用されていないもの）は引き続き拒否すること。既存テスト test_adoption_rejects_unreached_and_unknown_evidence_before_writing の期待値を要件に合わせて直し、Cを採録できるテストを追加すること。

修正後、変更範囲のpytest、kg_validate.py、Web UIへの影響があればnpmの検証を再実行し、PRへpushしてください。PR本文の設計説明も更新してください。
