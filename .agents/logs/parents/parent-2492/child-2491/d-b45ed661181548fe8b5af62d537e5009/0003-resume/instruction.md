## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261124T124209Z_8a4890ab19eb_fasthmr-public-assets",
      "name": "fasthmr-public-assets",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 1063.0,
      "log": "[local path]",
      "log_excerpt": "downloading feature_extractors/camerahmr.pth.tr (2210774984 bytes)\ndownloading fasthmr_ckpts/camerahmr_diffdecoder.pth.tr (678576471 bytes)\ndownloading feature_extractors/yolo11x-seg.pt (125090821 bytes)\ndownloading fasthmr_ckpts/vae.pth.tr (626670806 bytes)\nverified feature_extractors/yolo11x-seg.pt 125090821 4e53a5f5fd3ee2ae3361c62169c6bb3ed4ae251dd0e57606e230955aa52d919c\nverified body_models/J_regressor_h36m.npy 937168 c655cd7013d7829eb9acbebf0e43f952a3fa0305a53c35880e39192bfb6444a0\nverified fasthmr_ckpts/vae.pth.tr 626670806 218055d749b0a00118cfaf26efc4dac8b1c7c6840884e083b108d3da33c95b07\nverified fasthmr_ckpts/camerahmr_diffdecoder.pth.tr 678576471 ba9dcf7dfa3bed46c01fb814c25b88cece68919328ea7b757bbf30a20b1fcccc\nverified feature_extractors/camerahmr.pth.tr 2210774984 8d814cbf95b0efac89a939db055b6f67feacab0c85a43d3d307f49de313c9d4e\nverified 5 public assets, receipt outputs/fasthmr-setup/public-assets.json"
    }
  ]
}
