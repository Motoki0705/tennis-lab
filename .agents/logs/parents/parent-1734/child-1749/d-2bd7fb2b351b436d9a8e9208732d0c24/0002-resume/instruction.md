## 修正依頼（PR #2156、正式レビュー parent-1734 の結果 request_changes）

1. [major] pr-2156-1 `src/tennis_scene/pipeline/runner.py:37`: clip Aの完了callbackが公開エラー（入力改変の検出など）を送出すると、`_notified_clips.add` が実行されない。例外を捕捉して同じexecutorを再開すると、targetsを別のclip Bに限っても、冒頭の全clip通知でAのcallbackが再実行され、同じエラーでBへ進めない。
   - 通知の「試行済み」と「公開の成否」を分けて保持する。失敗したcallbackを暗黙に再通知しない。
   - 失敗したclipの状態は、呼び出し側（#1752の一括CLIが `continue_on_error` で使う）から読めるようにする。
   - 回帰テストを追加する: callback失敗→同じexecutorで別clipのtargetを再開→Bが完了し、Aのcallbackが再実行されず、Aは失敗として記録されている。
   - 単一clip入口とdataset入口で、公開失敗が従来どおり非0の終了・失敗markerになることも確かめる。

2. 証拠待ち: 「31 nodeの変更前後一致」が独立した比較であることを示す。
   - base `e561377ce` とHEADを、それぞれ空のstoreから別々に実行して比べるスクリプトと実行記録を用意する。
   - 示すこと: 各版の読込コードの版、code identityの固定方法、除外した項目とその理由、31 nodeそれぞれの実行[local path]、scene.json・artifact参照・scene metadata・annotation.jsonの比較結果。
   - 修正後のHEADでも同じ比較を取り直す。
   - 記録はknowledge（または、再現用のスクリプトをPR内のテスト支援に置いてもよい）に残し、PR本文から辿れるようにする。ローカル絶対パス・session IDを含めない。

修正後に対象HEADのCI（`Python` を含む）の成功を確かめ、新しいHEAD・テスト結果・比較記録の場所を報告する。
