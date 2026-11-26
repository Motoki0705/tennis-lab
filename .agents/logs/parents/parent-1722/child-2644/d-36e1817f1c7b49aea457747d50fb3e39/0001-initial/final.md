#2644 の実装を `codex/parent-1722-issue-2644-primary` の `5c89ec41be1599185196f4ca32b435353dc3e3a4` にコミットしました。

3方式すべてで、checkout の commit だけを変えた場合に `execute` と `load` が同じ成果物を再利用し、公開マーカーと計算時の commit が保持されることを確認しました。ソースや設定を変えると、`execute` は再計算し、`load` は明示的に拒否します。関連テスト113件と pre-commit が通過しました。旧形式の実行キーを持つ `event_detection` は移行時に一度 `execute` が必要です。

公開済み配布版の全チェックサム、69クリップの v3 読込、#2613 を一時的に統合した checkout での `post_train prepare` を照合するジョブを起動しました。照合が通ったら、実行証拠を添えてPRを作成し、CIへ引き渡します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261126T115531Z_a05a161f7458_i2644-published-inputs"]}}