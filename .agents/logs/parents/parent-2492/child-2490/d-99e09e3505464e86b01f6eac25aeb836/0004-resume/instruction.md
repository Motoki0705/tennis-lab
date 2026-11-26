## parent CLI が確認した待機結果

全ローカルコマンドが終了した。各jobの exit_code・log・log_excerpt を確認し、失敗や取消があれば原因を修正して再実行する。問題がなければ commit・push・CI待ち、または後続jobの登録へ進む。
親の判断が必要なら、理由と必要な判断を返す。

{
  "kind": "local",
  "jobs": [
    {
      "job_id": "20261124T132058Z_94a174870c92_sapiens2-runtime-onnx-1b",
      "name": "sapiens2-runtime-onnx-1b",
      "state": "done",
      "exit_code": 0,
      "duration_seconds": 383.783,
      "log": "[local path]",
      "log_excerpt": "Using Python 3.12.3 environment at: outputs/sapiens2/runtime\nRuntime 2.13.0+cu130 13.0 10.14.1.48.post1 0.7.2\nsapiens2-1b verified 696 checkpoint tensor names and shapes\nsapiens2-5b verified 968 checkpoint tensor names and shapes\n/usr/lib/python3.12/copyreg.py:99: FutureWarning: `isinstance(treespec, LeafSpec)` is deprecated, use `isinstance(treespec, TreeSpec) and treespec.is_leaf()` instead.\n  return cls.__new__(cls, *args)\n[torch.onnx] Obtain model graph for `PoseTopdownEstimator([...]` with `torch.export.export(..., strict=False)`...\n[torch.onnx] Obtain model graph for `PoseTopdownEstimator([...]` with `torch.export.export(..., strict=False)`... ✅\n[torch.onnx] Run decompositions...\n[torch.onnx] Run decompositions... ✅\n[torch.onnx] Translate the graph into ONNX...\n[torch.onnx] Translate the graph into ONNX... ✅"
    }
  ]
}
