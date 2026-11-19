# PR #2056 の正式レビュー指摘への対応

正式レビュー（gpt-6-astra / xhigh、run parent-2010-af6f0106ec7940f88d3b6f8725a9b67e）の判定は request_changes で、major が2件ありました。親は1件目を main と PR HEAD（2f7e559af）の比較で確認しました。main の `interrupted` ハンドラは例外を投げる前に `Runtime.inspect(runtime_dir, stop=True)` で runner を止めていましたが、PR ではそれが無くなり、例外が `subprocess.run` まで伝わって runner を SIGKILL します。2件目はあなたがコードで確かめてください。

## pr-2056-1（major）: `.agents/ops/dispatcher/service.py:169`

規則: 通常停止でrunnerの回収・復旧契約を壊さない

静的読解による指摘です。judge()がsubprocess.run()でrunnerを待っている間にSIGTERM/SIGINTを受けると、このInterruptedErrorがsubprocess.run()まで伝播し、その例外処理がrunnerをSIGKILLします。後段のinspect(stop=True)では既にsubreaperを失っており、agent_runtime.stop_lost_runner()は別途cleanup確認がない限りcleanup_complete=Falseにします。従来は例外を投げる前にrunnerへstopを送り回収できていたため、通常停止が手動回収必須へ退行します。runnerを強制終了する例外経路へ入る前に、ロック外の通常制御で停止・回収する構造へ変更してください。追加テストは実行中judgementへのシグナルを検査していません。

## pr-2056-2（major）: `.agents/ops/dispatcher/supervisor.py:1341`

規則: 最後のロック外I/O後にrunを再照合してから結果を保存する

静的読解による指摘です。current_job()でrun_idを照合した後、_finish()の親completed経路が_authorized()[local path]。この通信中にruntime/result.jsonが別runへ置換されても、Storeの比較対象はregistryとscheduleだけなので検出されず、古い終了証拠からjob・親をendedにして容量を解放できます。accept()もcurrent_job()後の_authorized()に同じ隙間があります。GitHub照合を終えてからrunを再検証し、以後は外部I/Oを挟まず適用・保存してください。既存のrun置換テストはinspect中の変更だけで、この後続通信中の変更を検査していません。

## 依頼

1. 2件とも、コードを読んで成り立つかを確かめる。成り立たない場合は、根拠（該当行と実行経路）を最終報告に書き、変更しない。
2. 成り立つものは PR #2056 の branch（`codex/dispatcher-triage-io`）で直す。
   - 1件目: シグナルを受けたときも、runner をロックの外で通常の手順で止めて回収し、そのあとで中断として扱う。シグナルハンドラの中でロックを取る処理や待ちを呼ばない方針は保つ。たとえば、ハンドラはフラグを立てるだけにして、runner の待ちを中断可能な形にする、などを検討する。cleanup_complete=False（手動回収が必要）になる経路を作らない。
   - 2件目: 最後のロック外 GitHub 通信（`_authorized()`[local path]`authorization()` など）を終えてから run（runtime/result.json の同一性を含む）を再照合し、そのあとは外部I/Oを挟まずに適用・保存する。`accept()` の同じ隙間も直す。
3. 回帰テストを足す。実行中の judgement にシグナルを送ったとき runner が通常停止・回収されること、後続の GitHub 通信中に run が置き換わったら古い結果を適用しないことを確かめる。テストは設計してから書く。
4. 関連テストと pre-commit を通し、push して CI の成功を確かめる。PR 本文の検証欄を更新する。

最終報告には、指摘ごとの判断（直した／成り立たない）、変更点、追加したテスト、実行したコマンドと結果、新しい HEAD を書いてください。
