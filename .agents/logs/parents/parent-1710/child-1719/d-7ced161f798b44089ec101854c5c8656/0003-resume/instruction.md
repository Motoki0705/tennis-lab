PR #1793 の統合レビュー（run parent-1710-53b9c7c8159b49a38c52ded58c3f4ad6、PRへ公開済み）が request_changes でした。major 1件を同じbranchで修正し、push後のCI結果まで確認してください。

pr-1793-1（jobs.py:433付近）: `store.restore()` が current.json を os.replace した後のfsyncで失敗すると、現行版は既に切り替わっているのに storage_error を返す。jobs側は公開済みrevisionを照合せずにerrorをjournalへ確定するため、同IDの再送が永続的に失敗し、`_recover()` もerror付きのrestore記録を照合対象から外すので再起動でも修復されない。
- 編集要求と同じく、undo（restore）でも、失敗を確定する前に公開済みのcurrent revisionを照合し、実際に切り替わっていれば成功として記録する。切り替わっていなければ失敗として確定する。判断できない場合は成功扱いにしない。
- `_recover()` も、error付きのrestore記録を含めて公開状態と照合する。
- undoのcurrent置換後（fsync段階）に障害を注入する回帰テストを追加し、(a) 同ID再送の結果、(b) 再起動後の状態、(c) 現行版が独立期待値と一致すること、を確かめる。修正前のコードで失敗することも確認する。
- 同じ「置換後に失敗を返す」パターンが、export・その他のstore操作の呼出しに残っていないかも確認し、あれば同じ方針で直す。

#1720 のworkerがこのbranchの旧HEADの上で作業中なので、HTTP入口・状態名・token受渡しの公開面は変えない。
最終報告: HEAD・CI結果、修正内容、回帰テスト（コマンドと結果、修正前に失敗することの確認）、同パターンの点検結果。目安は最大2時間。
