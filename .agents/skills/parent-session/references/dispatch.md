# dispatcherから起動された親

1. 起動指示末尾の `bootstrap_input` を読み、同じrepo・親番号・stateで `dispatcher accept --input` を実行する。候補・依存・予約時の根拠を照合する。
2. parent CLIの共通引数には `--state-dir <parent_state_dir>`、ワーカー操作には加えて `--dispatcher-state-dir <dispatcher_state_dir>` をサブコマンドの前に渡す。`launch_plan` の依存・空き枠を見て [ワーカー](workers.md) を選ぶ。
3. 終了・判断要求のイベントを受けたら、保存済みjobと成果を照合する。実行の継続が必要なら `continue-worker` を使う。
4. `workflow-report --spec` の出力をそのまま最終JSONにする。待機・失敗はversion 1で理由・再開条件・証拠を指定する。完了はversion 2に [全体のfinish](finish.md) を入れる。担当終了はsupervisorが行う。

単独Issueがマージで閉じた直後に親が中断したら、同じ親で `dispatcher resume` し、残りの受入・報告を終える。予約済み起動やstateに不整合がある場合は [復旧](recovery.md) を読む。
