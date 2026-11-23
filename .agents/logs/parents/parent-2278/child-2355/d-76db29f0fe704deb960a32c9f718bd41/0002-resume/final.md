commit `67468fd4efc9d3d855ab357ed21dc48310a997ec` で調査campaignを実装しました。

- 46-nodeの最小ScatterND再現、FP32比較、Slice/Clip/Concat書換えを追加。
- 実際のB=2/L=145失敗ONNXで、書換え前後の全5出力がONNX Runtime上でbit一致。
- CPU回帰58件成功、ruff成功、knowledge検証721 node・error/warning 0。
- TensorRT 10.14.1 release notesには同一不具合の記載なし。類似の[#3527](https://github.com/NVIDIA/TensorRT/issues/3527)は条件が異なり未確定。
- GPU campaignはqueueへ登録済み。完了後に結果集計、knowledge node、最終文書を作成します。

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204678452795146713_2026-11-22T04-25-47-204853286Z_363852_i2355-gvhmr-scatternd-campaign-1.job"]}}