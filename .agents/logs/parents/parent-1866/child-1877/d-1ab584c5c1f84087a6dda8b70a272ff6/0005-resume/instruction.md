PR #1883 の再レビュー（run parent-1866-9c85e1946672465ab51730b0e70e1dc3、head ac50f2c9b）で major 指摘が1件出た。前回の2指摘（pr-1883-1 前文・pr-1883-2 decision_id）は解消済みと判定されている。今回の指摘を修正してほしい。

指摘 pr-1883-1（今回のrunでのID）: `.agents/ops/parent/improvements.py` の `agreement()`（L600付近）。
`decide()` は承認時に対象issueの現在本文から `requirements: contract(body)` を固定するため、人間が起票後の本文を編集して範囲を変えた（例: 候補「取得と削除」→ 本文「取得のみ、削除は対象外」）うえで実行承認を記録できる。しかし `agreement()` はその承認snapshotではなく旧候補specの purpose/scope/out_of_scope/behavior/acceptance/stop_conditions から実行合意を生成する。結果として、人間が承認した範囲と agree が保存する人間合意の範囲が食い違い、`record_autonomous()` も旧範囲との一致を要求する。親もコードで確認し、指摘は成立すると判断した。

求めること:
- 承認した範囲そのものが実行根拠（agree の合意・autonomous 側の照合）へ渡ること。方式は次のどちらかを根拠付きで選ぶ: (a) 承認snapshot（承認時の本文の要件）から実行合意を生成する、(b) 本文が候補specの描画と契約上一致しない限り approve を拒否し、本文の変更は候補の新revisionとして記録・再公開させる。どちらでも、食い違いを黙って旧範囲で埋める静かなフォールバックは不可。(a) の場合、本文から必須の欄を取り出せないときは明示エラーにする。
- 回帰テスト: 起票後に本文で範囲を縮小して承認したケースで、agree が保存する合意（scope・acceptance 等）が承認内容と一致する（または (b) なら承認が拒否され、新revisionの承認後は新範囲で合意される）ことを E2E で固定する。修正前に失敗することを確認する。
- 既存の承認・再承認・撤回・前文保護・decision_id 照合のテストを壊さない。関連 suite と pre-commit を通し、PR #1883 に push する。PR本文の該当箇所も必要なら更新する。
- main は直近で取り込み済み。新たに main を merge しないこと（レビューの修正確認が base 取込みの全パス一覧でコメント上限を超えるため）。main と衝突した場合は merge せずに報告して止まる。

最終報告: 選んだ方式と理由、変更ファイル、修正前の失敗確認、実行したテストと結果、push後のHEAD、CI状況。
