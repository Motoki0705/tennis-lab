RT-DETRv4 L/X の `trt16-fp32attn` 診断を実装し、GPU ジョブを各1件投入しました。実物の ONNX で、各モデルの decoder 6層にあるスコア MatMul と Softmax だけが FP32 になり、次の MatMul への入力が FP16 に戻ることを確認しました。入出力仕様と外部重みの SHA-256 も元のグラフと一致しました。 [実装と再現手順](https://github.com/Motoki0705/tennis-lab/blob/0ed2bb1d7289ace7758c023bb01397acbfff22c4/knowledge/runs/run-meiji-player-model-comparison/rtdetrv4.md#fp32-decoder-attention-probe)

診断では前回と同じ4画像を使い、batch 1・2 それぞれで元の順序とペア内を入れ替えた順序を比較します。保存済み PT16 に対する生出力の差、BBox の一致数、疑似GT Recall を記録します。GPU の結果を受けて、改善したモデルの全6,844画像評価と速度計測へ進みます。

pytest 87件（追加5件、8.70秒）、pre-commit、knowledge 検証が通過しました。`codex/parent-2492-issue-2487-rtdetrv4` の HEAD `0ed2bb1d7289ace7758c023bb01397acbfff22c4` を push し、CI を起動しました。 [CI](https://github.com/Motoki0705/tennis-lab/actions/runs/45633433760)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204383399993249512_2026-11-25T14-23-20-006750487Z_1262166_i2487-rtdetrv4-l-trt16-fp32attn-v1.job","0-8204383399731915466_2026-11-25T14-23-20-268084533Z_1262198_i2487-rtdetrv4-x-trt16-fp32attn-v1.job"]}}