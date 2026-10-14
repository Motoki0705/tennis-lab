# 版付きの一括編集と引継ぎ

`RevisionStore` は元予測から不変revisionを作り、taskの意味検証を通った一括編集だけを
CASで現行版へ切り替える。登録済み新規clipのprediction receiptと、根拠付きレビュー方針を持つ
production planが必要である。previewや別campaignの予測は編集開始に使わない。

`ReviewContext` がplan・task・clip manifest・prediction receiptを固定する。
`ReviewAdapter` は初期payload作成、全payloadの検証、編集batchの適用、教師採用の検証を実装する。
baseは球状態・bbox・ID・教師maskの意味を解釈しない。画像提示の共通APIは
[images](../images/README.md)を参照する。

1. `initialize(context, adapter)` は元予測を変更せずrevision 0を保存する。同じcontextの再開は
   現行版を返し、初期payloadに戻さない。
2. `apply(EditCommand, adapter)` は期待するrevision、編集入力、actorの根拠、理由を固定する。
   adapterは変更全体と、確認済み実観測・端点・イベント/ID境界・派生ラベルを検査してから
   新payloadと小さな `ChangeSummary` を返す。途中失敗で一部の行だけを公開しない。
3. 新payload/revisionを不変保存し、headのCAS更新とedit receiptの全文を同じtransactionで保存する。
   同一command hashの再適用は元のreceiptを返す。commit直後のファイル保存失敗もDB行から復元できる。
   別の編集が先に進んだ場合は `RevisionConflict`、所有lockが使用中なら `BlockingIOError` を返す。
4. `seal(revision, adoption, adapter)` はtaskによる画像確認・方針・教師maskの検証と、共通の
   clip/版/予測/policy参照照合を行う。`ReviewEvidence.prediction` はprediction receiptを指す。
   採用後の新しい編集を拒否し、教師の上書きやモデル更新による再注釈を起動しない。

通常の編集結果は変更範囲・件数・最大32行のbefore/after例・未解決件数・receipt参照に絞る。
詳細はcommandと各revisionのpayloadから辿れる。現行notesを解消しても旧payloadの記録は残る。
taskの具体的な保護条件や画像確認の意味はadapterが実装し、汎用の成功flagでは代用しない。

## 引継ぎ

[`build_handoff()`](../handoff.py) は現行revision、最大64frameの範囲、最大64行のtask要約、
確認/未解決の範囲、event/track境界、次操作、採用参照、最大20件の編集参照を保存する。
taskは `summarize()` で正本payloadから要約を導出し、続きのrow cursorと残件数を返す。
`total_rows` は指定scope内の総行数、`remaining_rows` は今回のページの後に残る行数である。
全体JSONをプロンプトへ再投入する必要はなく、正本はhash参照で保持する。

rowの続きには `after_row` と `expected_revision` を渡す。版が進んだ場合は再取得を求め、
旧ページと新ページを混在させない。`previous_revision` を指定すると編集command参照を新しい順で返す。
履歴が一ページに収まらない場合は `history_cursor` を `changes_since()` へ渡して続ける。
画像を確認した範囲と、位置・状態が未解決の範囲は独立に保存する。
「画像は見たが位置が分からない」frameは両方に含まれ得る。観測ごとの部分状態はtask行へ保持する。

これらは親側のAPIである。workerからの操作受付、task用CLI、実際のball/player状態と画像確認方針は
各task adapterおよびcomposition rootへ接続する。合成adapterのテストは共通処理の検証であり、
実動画の注釈品質や実taskの採用判定を検証したことにはならない。
