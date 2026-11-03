# レビュー用の取得と隔離CPUテスト

正式レビューと全体調査が共有する実装。全体調査の `opsreport` は互換入口と台帳adapterを持ち、
正式レビューの `review_server` は複数の固定commitとrunごとのjournalを使う。

- `retrieval*`・`mcp_server`: 固定commitのgit読取、GitHub GET、ページ境界・切詰め・取得証拠。
- `reproduction`: 許可された既存CPUテストだけを使い捨てcloneで実行する。
- `repro_isolation`・`repro_linux`・`repro_sandbox`: Linux namespace/cgroupによる強制と実機probe。
- `records`: 記録先のProtocolと台帳に依存しないjournal。
- `review_server`: requestで固定したcommitごとの取得入口とテスト許可リスト。

隔離実行は最大15分・memory 4GiB・swap 0、GPU・外部network・credentialsを隠す。
サービス内は同時1件で、通常pytestには共通ロックを課さない。
OSの実機probeが有効でなければ `cpu_tests` を公開せず理由を返す。
node IDは親が固定commitに存在するCPUテストを選び、任意コマンド・テスト生成・依存追加は受け付けない。
probeと許可リストの準備手順は [全体調査の隔離実行](../../ops-report/README.md) を参照。
