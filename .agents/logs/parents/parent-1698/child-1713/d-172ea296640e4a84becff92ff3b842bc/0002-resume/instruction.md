PR #1769 の統合レビューで major の指摘が1件ありました。修正して同じPRへpushし、CIの成功を確認してください。

レビューのrunは、通信の一時的な切断のため形式上 failed として記録されました。ただし、レビュー担当は最後まで判定を出しています。親が該当コードを読み、指摘が正しいことを確認しました。

指摘（`src/tasks/annotation/local_cli.py` 47行付近、静的解析による指摘）:
`execute_local` は、`workflow/annotation-policy.json`（benchmarkでは `benchmark/policy.json`）がなければ `DEFAULT_POLICY` を使います。次の手順で問題が起きます。
1. `DEFAULT_POLICY` と異なるpolicyを指定する。
2. `SceneAnnotationWorkflow.annotate` が動画の登録を終えた後、controllerを初期化する前に中断する。この時点で `scene-workflow/request.json` には指定したpolicyが保存済みだが、`annotation-policy.json` はまだない。
3. 復旧CLIの `status` を呼ぶだけで `DEFAULT_POLICY` が選ばれ、`LocalAnnotationWorkflow` の初期化で不変ファイルとして保存される。

その後、元のscene workflowは指定したpolicyを書こうとして「immutable document already has different content」で停止します。`run` や `tick` なら、誤った方針で注釈が始まります。これは静かなフォールバックの禁止に反します。

求める修正:
- 探索方針は、登録時に固定した値だけを使う。保存済みのscene requestや登録記録など、正本となる固定記録から復元・照合する。欠けている場合に既定値で置き換えない。正本が読めないときや食い違うときは、明示的なエラーで止める。
- `DEFAULT_POLICY` を使ってよいのは、利用者が方針を指定しなかった登録の時点だけにする。その場合も、その値を登録記録へ固定する。復旧・status・tick・runの各経路から既定値を選ぶ分岐をなくす。benchmark経路も同じ規則で確認する。
- 回帰テストを追加する。既定と異なるpolicyで登録した直後（controller初期化前）に中断し、復旧CLIの `status`、`tick`、`run` のいずれかを呼んでも、指定したpolicyが使われる（または明示エラーになる）こと。元のworkflowを再開すると、同じpolicyで完了まで進むこと。修正前のコードでこのテストが失敗することも確認する。

変更範囲のテストは `-n 0` または `-n 2` で、対象に絞って実行してください。全体はCIに任せます。最終報告には、修正commit、追加したテスト名とその結果、CIの結論を書いてください。
