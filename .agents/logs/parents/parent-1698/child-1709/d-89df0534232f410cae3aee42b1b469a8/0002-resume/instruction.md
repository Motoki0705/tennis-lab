PR #1766 の統合レビューで major の指摘が1件ありました。修正して同じPRへpushし、CIの成功を確認してください。

指摘 pr-1766-1（`src/tasks/annotation/recovery.py` 80行付近、静的推論による指摘）:
`recover_annotation` は、継続判断を保存してworkとserviceのlockを解放した後に `pool.authorize_retry` を呼びます。そのため、timeoutの回収と並行して動いていた `stop()` が、その間に同じattemptを `cancelled` へ確定できます。`authorize_retry` は `RETRYABLE_STATES` に含まれる `cancelled` も受理するので、取消が `ready` に上書きされ、次のtickで自動的に再起動されます。自動継続は `timed_out` に限り、取消は明示的なretryまで保持するという契約に反します。

求める修正:
- 自動経路では、許可の確定を「同じattemptで、かつ状態が `timed_out` のときだけ」に限定する。状態の照合と `ready` への更新は同じlock・同じ条件付きUPDATEの中で行う。例: `authorize_retry` に、受理する状態を呼出元が指定できる引数を足し、自動経路は `timed_out` だけを渡す。手動の `RetryWorker` は従来どおり全ての `RETRYABLE_STATES` を受理する。
- 保存済みの判断を再適用する場合も、現在の停止状態が `timed_out` であることを照合する。状態が `cancelled` などに変わっていたら、許可せずに現状をそのまま報告する。保存済みの判断ファイルは書き換えない。
- 回帰テストを1件追加する。判断の保存と許可の間に `stop()`（またはそれと等価な取消の確定）が割り込み、attemptが `cancelled` になるケースで、状態が `ready` にならず、次のtickでworkerが起動しないこと。既存の別attemptへの交代テストとは別に用意する。

変更範囲のテストは `-n 0` または `-n 2` で、対象に絞って実行してください。全体はCIに任せます。最終報告には、修正commit、追加したテスト名とその結果、CIの結論を書いてください。
