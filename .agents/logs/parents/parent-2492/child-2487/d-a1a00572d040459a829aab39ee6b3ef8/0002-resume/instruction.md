## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261124T113947Z_a94715bdcac9_edgecrafter-assets-and-cpu-smoke",
      "name": "edgecrafter-assets-and-cpu-smoke",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 409.884,
      "log": "[local path]",
      "log_excerpt": "ecdet-x-coco outputs/player-model-comparison/assets/ecdet_x.pth 200304616 bytes sha256=c87394be021b573fd8e25969b85a8670127120c9d0209ac7807fbadae1a5cec6\necpose-l-o365 outputs/player-model-comparison/assets/ecpose_l_o3652coco.pth 138759372 bytes sha256=091ad72c92ee5de3383176cbae8e8836e9cacf2b9fbec94c11c855fbfe35343c\necdet-x-o365 outputs/player-model-comparison/assets/ecdet_x_o3652coco.pth 200357826 bytes sha256=be9c77cd4def8aadd0cc679944ba57412f363de2d44f7c6b900289e61316314c\necpose-x-o365 outputs/player-model-comparison/assets/ecpose_x_o3652coco.pth 206712140 bytes sha256=2d79c0fd21831e4f3e6d5ba7bfa10b5e6e8ecb2641180299d7a1bd968b33f4f6\necdet-l-coco pt32 cpu forwarded 2 fixed images to outputs/player-model-comparison/ecdet-l-coco/smoke-pt32.json\necdet-x-coco pt32 cpu forwarded 2 fixed images to outputs/player-model-comparison/ecdet-x-coco/smoke-pt32.json\necdet-l-o365 pt32 cpu forwarded 2 fixed images to outputs/player-model-comparison/ecdet-l-o365/smoke-pt32.json\necdet-x-o365 pt32 cpu forwarded 2 fixed images to outputs/player-model-comparison/ecdet-x-o365/smoke-pt32.json\nnum_body_points: 17\necpose-l-o365 pt32 cpu forwarded 2 fixed images to outputs/player-model-comparison/ecpose-l-o365/smoke-pt32.json\nnum_body_points: 17\necpose-x-o365 pt32 cpu forwarded 2 fixed images to outputs/player-model-comparison/ecpose-x-o365/smoke-pt32.json"
    }
  ]
}
