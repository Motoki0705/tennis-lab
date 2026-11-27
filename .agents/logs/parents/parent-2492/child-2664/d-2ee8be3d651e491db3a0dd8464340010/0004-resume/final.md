GPU 照合は RF-DETR-L・RTMPose-X の PT16・TRT16 全4条件で合格しました。固定入力で、比較時の実装との座標・score の最大差はすべて 0 でした。

採用済みの共有コート校正を使った4カメラ実行では、369フレームを処理し、GVHMR とイベント検出を含む48段が完了しました。成果物のマニフェストを全件 SHA-256 照合し、シーンアーカイブを読み戻して `status=ok` を確認しました。検証した HEAD は `851d3373b8ffbaf067fa8cc97a4f0fc249041e76` です。結果を [#2664](https://github.com/Motoki0705/tennis-lab/issues/2664#issuecomment-7206315590) に記録しました。

全体の設定一覧テストと設定監査を起動しました。結果の確認、push、PR 作成が残っています。#2470 の3D追跡から公開までの通し確認は #2487 に引き継ぎます。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261127T001637Z_7bc7ce85847b_i2664-full-config"]}}