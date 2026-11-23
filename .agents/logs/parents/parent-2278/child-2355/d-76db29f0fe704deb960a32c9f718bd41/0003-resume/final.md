初回campaignの有効な結果を確認しました。

- ScatterND単体はB=2 FP16/FP32、最適化レベル0/3すべてbuild成功。
- ScatterND除去後の実モデルは依然 `build_serialized_network` 内でSIGSEGV。
- 実モデルFP32のPyTorch・元ONNX・書換えONNXは一致し、書換え前後は全5出力bit一致。
- 原因仮説をScatterNDからTransformer/attention融合へ修正。
- subgraph診断をcommit `00e4a0136f23a6db3b8c670a5f9b3b743b195595` で追加し、queueへ投入済みです。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204664340393763245_2026-11-22T08-20-59-606236754Z_2290638_i2355-gvhmr-subgraph-diagnosis-1.job"]}}