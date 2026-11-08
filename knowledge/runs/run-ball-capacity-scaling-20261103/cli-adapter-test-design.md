# CLI再開adapterの検証設計

守る契約: 保存済みcontroller/source/lock/plan/preparedには触れず、mounted_driveを専用OAuthの同じDrive treeへ接続する。control stageはreceiptとmarkerだけを取得し、recover.py自身が全snapshotを検証する。datasetは固定distributionの全件をSSDへ転送後、サイズ・hashlibとCPythonの独立SHA256・membershipを検証する。files receiptを公開してからcomplete progressを公開する。

見逃したくない不具合: aliasが別treeを指す、重複/unsafe path、異なるmanifest、同サイズ破損、欠落/余分なpayload、remote inventoryの差、失敗後にcomplete receiptが残ること。秘密設定はstdoutへ出さない。

最軽量の層: knowledge側の標準ライブラリadapterをunit testで検証。実rclone/Drive/全datasetとsnapshotの接続はVMで実測する。既存 tests/unit/tasks/ball_detection/training/test_sequential_recovery.py の最新破損・曖昧候補・既存状態保護のテストを再利用する。snapshot restoreは tests/unit/utils/artifact_store/test_snapshots.py を再利用する。

予算: 小さなローカルfixtureのunit 12件以下、各5秒以内。実データはVMだけで2925 payload/157028755547 bytesを一巡検証。GPU学習はqueueのみ、受入はd512-l4の5500→6000保存全件検証まで。

VMでの読み取り専用qualificationは verify_cli_resume.py に分離する。元inputsのprepared全6件、source全件、controller/prior/report/theory/benchmark、model/budget/runtimeを固定計画と比較し、checkpoint自身のstrict resume検証、AdamW全momentの有限性、CPU/CUDA/Python/NumPy RNG、sample位置、連続更新ログprefixとloss/gradを検証する。安定保存時は同じsnapshot storeでDriveから独立復元し、markerと全payloadの検証を再実行する。checkpoint/weightはPCへ回収しない。

実測: 新規adapter 10件は0.29秒、全件5秒以内。既存復元/snapshotを含む初回43件は12.11秒で成功。既存controller再開の最初のcaseのみ6.42秒（import/初期化込み）、新規caseの重いfixtureはない。新規のremote前後差分2件を追加後、adapter10件を再実行して成功した。

実VMで追加発見: bootstrap repoの git ls-files --others が登録worktreeをディレクトリとして返し、現行Colab CLIのsource収集が read_bytes() で失敗する。VMのprivate Git info/excludeへ登録済みの指定worktreeだけを明示追加する。tracked sourceは変更しない。実Gitの小さなrepo/worktreeで、再現→除外→実験側tracked sourceが残ることを検証するunit 1件を追加（合計11件、5秒以内）。
