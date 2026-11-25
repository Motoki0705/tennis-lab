# #2487 edgecrafter: decoder self-attention を FP32 にした plan で全体を測り直す（親の判断）

前回までの報告（`e9a12d172` まで）を受け取った。PT16 の速度計測と、TensorRT の切り分けを承認する。

## 判断
診断 v3 で、decoder self-attention の score MatMul 4つと Softmax だけを FP32 にした ECDet-L COCO の plan は、保存済み PT16 との一致が大きく改善した（batch 1: 30/68 → 56/68、score ≥0.25 では 3/12 → 10/12）。FP16 の attention score が主な原因である可能性が高い。前回の指示（直せるなら engine を作り直して、精度と速度を測り直す）に沿って、全体の評価へ進める。

## やること
1. 同じ書き換え（decoder self-attention の score MatMul と Softmax だけを FP32 にし、ほかは FP16 のまま）を、6重みすべてに当てた TensorRT plan を作る。対象は ECDet L/X × COCO/O365 と、ECPose L/X O365。graph の構造が違って同じ場所を特定できない重みがあれば、その重みは理由を付けて「適用不可」と記録する。黙って別の書き換えにしない。
2. 新しい条件の名前は、元の TRT16 と区別する（例: `trt16-fp32attn`）。元の TRT16 の結果は消さずに残し、「変換の不具合の疑い（FP16 の decoder attention）」と注記する。
3. 新しい plan で、固定の全6,844画像の精度を、元の TRT16 と同じ採点器・同じ batch で測る。続けて、main の harness（parity 未判定）で batch 1・2 の速度を測る。PT16 の速度計測と同じホストのスレッド設定にする。
4. batch の並び順で結果が変わる効果が残っていることも、記録に書く。精度は batch 1 で測るので、比較表に載せるのは batch 1 の精度とする。
5. GPU は training queue 経由で、小さい job に分けて入れる。いま queue は空いている。
6. PT32 は推論しない。比べる相手は保存済みの PT16。

## 終わり方
テストと pre-commit を通して push する。最終回答には、作った plan（適用不可の重みがあれば理由も）、入れた job、push した HEAD を書き、queue を待つ `PARENT_RESULT`（`job_ids` 付き）を返す。
