PR #1800 の統合レビュー（run parent-1710-6a8c1d0478c5421d80b7605358ec5382、PRへ公開済み）が request_changes でした。annotationテストの同梱と、隔離・公開境界・CSP・export は妥当と判定されています。major 3件を同じbranchで修正し、push後のCI結果まで確認してください。

1. pr-1800-1（web/app.js:478付近）: undoの通信が公開確定前に切れ、最初の履歴取得がcommitより先に終わると、履歴に要求IDがないだけでpendingを削除してしまう。serviceのrestoreは別スレッドで続き、後で公開されてもUIは旧版と「restoring」表示のまま止まる。未掲載を終端として扱わず、要求IDを保持したまま照合を続ける（編集要求と同じく、状態を取得できる入口があればそれを使う）。確定できない場合は、明示的な未確定状態と回復操作を出す。再接続を理由に再送しない。ブラウザ検証に、undoの応答消失とcommitの遅れの競合を加える。
2. pr-1800-2（web/reader.js:33付近）: 内容IDが入れ子の場合、文書順で最初に画面上端を含む外側blockがanchorになる。内側の節への追記後に、未変更の後半の節がずれる。最も内側で画面上端を含むblock（またはそれに準じる安定した要素）をanchorにする。入れ子のarticle/sectionでの検証を加える。
3. pr-1800-3（web/reader.js:25付近）: `querySelectorAll` がblock自身を含まないため、`<details data-content-id=...>` 自身のopen状態を保存・復元できない。block自身も対象に含める。block自身がdetailsのケースの検証を加える。

各修正は、修正前のコードで新しい検証が失敗することを確認する。HTTP入口・状態名・token受渡し・postMessageの許可内容は変えない。
#1721 のworkerがこのbranchの旧HEAD（a0f2427）の上で作業中です。公開面を変えた場合は最終報告に列挙する。
最終報告: HEAD・CI結果、各指摘への対応、回帰検証（コマンドと結果、修正前に失敗することの確認）。目安は最大2時間。
