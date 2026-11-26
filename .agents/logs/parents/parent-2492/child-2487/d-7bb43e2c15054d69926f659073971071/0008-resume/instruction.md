# #2487 rtdetrv4: decoder self-attention を FP32 にした plan を試す（親の判断）

前回の報告（`fef762822`、TRT16 の Recall@0.50 が L 0.00%・X 0.03%、原因未特定）を受け取った。診断と記録を承認する。

## 参考: EdgeCrafter で見つかったこと
同じ DETR 系の ECDet-L COCO で、decoder self-attention の score MatMul 4つと Softmax だけを FP32 にした TensorRT plan を作ったところ、保存済み PT16 との一致が大きく改善した（batch 1: 30/68 → 56/68）。FP16 の attention score が原因である可能性が高い。手順と書き換えのコードは EdgeCrafter 担当の branch（`codex/parent-2492-issue-2487-edgecrafter`、`knowledge/runs/run-meiji-player-model-comparison/EDGECRAFTER.md` と `edgecrafter_trt_probe.py` の `--fp32-decoder-attention`）にある。必要なら取り込むか参照してよい。ただし、相手の branch は編集しない。

## やること
1. RT-DETRv4 L/X の ONNX で、decoder self-attention の score MatMul と Softmax に当たる場所を特定する。そこだけを FP32 にし、ほかは FP16 のままにした plan を作る。特定できなければ、理由を付けて「適用不可」と記録する。黙って別の書き換えにしない。
2. まず、前回の診断と同じ固定4画像で、保存済み PT16 と raw heads・decoded boxes を比べる。batch 1・2 と、並びを入れ替えた順で比べる。
3. 一致が大きく改善したら、全6,844画像の精度（batch 1、元の TRT16 と同じ採点器）と、main の harness での batch 1・2 の速度を測る。改善しなければ、そこで止めて記録する。
4. 条件の名前は元の TRT16 と区別する（例: `trt16-fp32attn`）。元の TRT16 の結果は消さずに残す。
5. GPU は training queue 経由で入れる。PT32 は推論しない。

## 終わり方
テストと pre-commit を通して push する。最終回答には、結果、入れた job、push した HEAD を書き、待つものに合わせた `PARENT_RESULT` を返す。2の結果を見てから3の job を入れる場合は、2の job を待つ `waiting` / `queue` で返してよい。
