## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261124T135319Z_b1fd89a7199a_edgecrafter-rescore-2495",
      "name": "edgecrafter-rescore-2495",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 36.856,
      "log": "[local path]",
      "log_excerpt": "ecdet-l-coco pt16 scored 6844 images. Affected aggregates []\necdet-l-coco pt32 scored 6844 images. Affected aggregates []\necdet-l-coco trt16 scored 6844 images. Affected aggregates []\necdet-l-o365 pt16 scored 6844 images. Affected aggregates []\necdet-l-o365 pt32 scored 6844 images. Affected aggregates []\necdet-x-coco pt16 scored 6844 images. Affected aggregates []\necdet-x-coco pt32 scored 6844 images. Affected aggregates []\necdet-x-o365 pt16 scored 6844 images. Affected aggregates []\necdet-x-o365 pt32 scored 6844 images. Affected aggregates []"
    }
  ]
}
