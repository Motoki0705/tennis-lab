# CLI再開adapterの検証設計

守る契約: 保存済みcontroller/source/lock/plan/preparedには触れず、mounted_driveを専用OAuthの同じDrive treeへ接続する。control stageはreceiptとmarkerだけを取得し、recover.py自身が全snapshotを検証する。datasetは固定distributionの全件をSSDへ転送後、サイズ・hashlibとCPythonの独立SHA256・membershipを検証する。files receiptを公開してからcomplete progressを公開する。

見逃したくない不具合: aliasが別treeを指す、重複/unsafe path、異なるmanifest、同サイズ破損、欠落/余分なpayload、remote inventoryの差、失敗後にcomplete receiptが残ること。秘密設定はstdoutへ出さない。

最軽量の層: knowledge側の標準ライブラリadapterをunit testで検証。実rclone/Drive/全datasetとsnapshotの接続はVMで実測する。既存 tests/unit/tasks/ball_detection/training/test_sequential_recovery.py の最新破損・曖昧候補・既存状態保護のテストを再利用する。snapshot restoreは tests/unit/utils/artifact_store/test_snapshots.py を再利用する。

予算: 小さなローカルfixtureのunit 12件以下、各5秒以内。実データはVMだけで2925 payload/157028755547 bytesを一巡検証。GPU学習はqueueのみ、受入はd512-l4の5500→6000保存全件検証まで。
