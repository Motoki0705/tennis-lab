# デスクトップ移管adapterの検証設計

- 契約: 固定commitの学習source・lock・PyTorch、元のrecipeとmanifest bytes、全model/AdamW/RNG/sampleを維持する。パスalias・永続化先・実GPUとallocator上限だけを明示して変える。
- 検出対象: source改変、weights-only、recipe変更、最新世代の取り違え、異なる教師/pack/augmentation、部分的なデータ取得を拒否する。元成果物を変更せず③だけを実行する。
- 層: adapterの局所拒否をCPU pytestで確認し、固定版の実runnerによるCPU統合テストを再利用する。GPU可否は共有queue内の実checkpointから5501以降を同一T64・microbatch・BF16・compile設定で実行して測る。
- 再利用: 固定版 `tests/integration/tasks/ball_detection/test_query_control_training.py` と `knowledge/runs/run-ball-query-controls-20261111/test_control.py`。model・optimizer・段境界・中断再開とpack照合を既存テストで守る。
- コスト: 固定版3条件とcontrollerの既存10ケースを一度実行し実測する。追加CPUテストは少数の小さなfixture。実GPU probeは2更新を予定し、上限とホストRAMを監視する。probeは別出力で本番checkpointを更新しない。失敗時は原因を保存し、自動再試行しない。

probe後に判明したsubreaperのzombieに対しては、queueのPGID検査を迂回せず、
systemd側でruntime終了イベントをpidfdで待ってから登録する。
実子process・pipe・pidfdを使うCPUテストで、process終了前にはqueue登録しないこと、
終了後のcwd・all予約・冪等keyを確認する。terminal eventはqueueのdone/failedを
根拠とし、Drive通知失敗を成功にせず自動再試行しないことを小さなfixtureで検証する。

本番の起動受入は固定runnerの実成果物を入力に検証する。旧基準との全log行照合、
6k validationのteacher/frame identity、checkpoint・named AdamW・RNG・sampleの
独立readback一致、後半1/600/601の実LRとfreezeを検査する。
件数だけの一致やreceiptの存在だけでは合格にしない。待機はinotify/pidfdで行い、
指定の受入点または失敗・process終了で検査processを終了する。
