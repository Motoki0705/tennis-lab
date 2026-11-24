ECDet 4種の PT32・PT16 で、6,844画像・26,276選手BBoxを採点しました。PT16 は4種とも PT32 との差分基準を満たしました。ECDet-L COCO の TensorRT は全画像で実行できましたが、人物行の一致率が66.4%で、基準の98%を下回りました。同じ ONNX を CPU で実行した診断では、確認した2フレームの24行すべてが PT32 と一致しました。[変換と診断の記録](https://github.com/Motoki0705/tennis-lab/blob/8e797063b2fe9a62991642f969fde8eea5cd9e8d/knowledge/runs/run-meiji-player-model-comparison/EDGECRAFTER.md)

更新された #2495 採点器と #2496 harness を取り込んだ `codex/parent-2492-issue-2487-edgecrafter` の HEAD `8e797063b2fe9a62991642f969fde8eea5cd9e8d` で、pytest 47件が通過しました。先に再採点した ECDet-L COCO の PT32 は集計値が一致しました。保存済み9条件の再採点をローカルジョブに登録しました。[再採点ログ]([local path])

ECPose の本採点は統合側の recipe と同じ外接矩形の10%拡張を使います。ECPose 全条件の採点、TensorRT の差分調査、#2496 の固定入力を使った速度・VRAM 計測が残っています。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"local","job_ids":["20261124T135319Z_b1fd89a7199a_edgecrafter-rescore-2495"]}}