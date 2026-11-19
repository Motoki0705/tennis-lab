PR #2085 の統合レビュー（run parent-1893-269d4c5ae7964e529b16ef1a83efcd0f、request_changes）で major 2件が出た。どちらも妥当と判断した。修正してほしい。

### pr-2085-1 .agents/ops/lib/agent_runtime.py:25（major）
静的解析で確認しました。Pythonはguardを実行する前に入口ファイル全体を読み込み・compileします。この858行のagent_runtime.pyやparent_review.py、merge_gate.py等は実処理も同じファイルにあるため、git-syncの排他lock待ちの間に入口と依存moduleが更新されると、解放後に「旧入口の実処理＋新依存module」で動きます。依存APIの変更で元のImportError/TypeError事故が再発し得ます。薄いbootstrapからlock取得後に実処理を読み込む構成にしてください。追加された更新先行テストは薄いparent入口の依存先だけを更新しており、入口自身の旧コード残存を検出しません。入口と依存先を同時更新する回帰ケースが必要です。

### pr-2085-2 .agents/ops/lib/checkout_lock.py:91（major）
静的解析で確認しました。保護対象外の直接dispatcher管理CLIには、registry lock保持中にRuntime.inspect()からagent_exec.shのstatus/stopを呼ぶ経路があります（dispatcher/service.pyのstop_judgement・recover_judgement、supervisor.pyのreconcile）。終了済み・中断済みjobの記録が残る状態でgit-syncが先に排他checkout lockを取得すると、管理CLIはregistryを保持したまま新しいrunnerの共有lock待ちになります。一方git-syncはMaintenance.pause()からdispatcher statusを呼び、同じregistry待ちになるため循環待機します。直接実行は停止し続け、systemd経由でもタイムアウトまで更新とCLI起動を塞ぎます。管理CLI側もregistry取得前に保護する等、statusの保守用途を維持しつつ取得順を揃えてください。既存git-syncテストはdispatcherをfakeに置換しており、この実経路の競合を検出しません。

## 親の指示
- pr-2085-1: 保護対象の入口は、lock取得前に読み込まれるファイル（Pythonの入口script・`__main__`、bashのscript）を、lock取得とその後の実処理の読み込みだけを行う薄いbootstrapにする。実処理はlock取得後にimportする（例: 入口ファイルは bootstrap だけにし、本体を別moduleへ移す）。入口と依存moduleを同時に更新する回帰ケース（lock待ちの間に入口ファイル自体と依存先の両方を新版に差し替え、解放後に新版の入口＋新版の依存だけで動くこと）を、16入口の代表と最も大きい入口（agent_runtime.py・parent_review.py・merge_gate.py）で検証する。全16入口が薄いbootstrapであることを機械的に検査するテスト（例: 入口ファイルがguard以外のproject importや実処理を持たない）も置く。
- pr-2085-2: lockの取得順を全経路で一つに揃える（checkout lock → registry lock の順など）。dispatcher の管理CLI（stop_judgement・recover_judgement・supervisor の reconcile など、registry保持中に agent_exec.sh の status/stop を呼ぶ経路）を含め、registryを保持したまま checkout lock を待つ経路をなくす。git-sync の Maintenance.pause() が dispatcher status を呼ぶ経路も同じ順序に従わせる。status/stop の保守用途（更新中でも停止できること）は維持する。fakeを使わず実際の dispatcher のregistry lockと agent_exec の経路を使う競合テストで、循環待機が起きないこと（時間制限付き）を確かめる。
- 他に同じ型（lock取得前の先読み、取得順の逆転）が残っていないか、入口一覧と lock を取る全箇所を調べ、結果をPR本文に書く。
- 修正を外すと新しいテストが失敗することを確かめる。関連テストと pre-commit を通し、通常の push（force pushしない）、PR本文の更新、対象HEADのCI確認まで進める。pytest の並列は `-n 4` 以下。systemd unit 操作・git-sync のON・メインcheckoutの変更はしない。作業時間の目安は最大5時間。
