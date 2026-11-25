再投入された診断2件を検証し、#2487 に結果を記録しました。

- **ECDet-L O365**は同じ画像順で生出力が一致しました。batch 2 の画像順を反転すると、IoU 0.50 以上で対応した人物BBoxは PT16 で254/254件、旧 TRT16 で199/243件でした。
- **ECDet-L COCO**の診断用 plan では、画像順を反転しても観測した TopK・Gather 出力は一致し、最終 logits の絶対差 P95 は0.5664でした。診断用 plan と旧 plan は同じ画像順でも93/241件の対応にとどまり、旧 TRT16 の精度低下は変換の不具合が疑われる暫定結果です。原因は未特定です。[診断記録](https://github.com/Motoki0705/tennis-lab/issues/2487#issuecomment-7164031943)

decoder の self-attention でスコア計算と Softmax を FP32 にした診断を1件投入しました。ONNX CPU では固定画像1枚で保存済み PT16 の68/68件と IoU 0.50 以上で対応し、TensorRT での結果を待ちます。旧速度24条件はホストスレッド2本、DINO 基準は4本だったため、stage wall time の同条件比較には再計測が必要です。

`codex/parent-2492-issue-2487-edgecrafter` の `9fa9b13a80ec8bdd80dddbaa408f39e094e2a7e5` を push しました。pytest は追加1件を含む88件が成功し、pre-commit とナレッジ検証も通過しました。[起動したCI](https://github.com/Motoki0705/tennis-lab/actions/runs/45625368485)

PARENT_RESULT {"schema_version":1,"state":"waiting","wait":{"kind":"queue","job_ids":["0-8204386175902366411_2026-11-25T13-37-04-097633588Z_1139308_i2487-ecdet-l-coco-trt16-fp32-decoder-attention.job"]}}